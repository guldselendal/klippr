"""
Unified LLM provider interface supporting multiple API providers.
Supports: Ollama, OpenAI, Gemini, DeepSeek
"""
from typing import Optional, List, Dict
import os
from dotenv import load_dotenv
from prompt_utils import STOP_SEQUENCES, TEMPERATURE
from llm_concurrency import LLMConcurrencyLimiter

load_dotenv()

# Provider clients (lazy initialization)
_ollama_client = None
_openai_client = None
_gemini_client = None
_deepseek_client = None


def get_ollama_client():
    """Get Ollama client with connection pooling"""
    global _ollama_client
    if _ollama_client is None:
        try:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            
            # Create session with connection pooling and retry strategy
            session = requests.Session()
            
            # Configure retry strategy for transient errors
            retry_strategy = Retry(
                total=2,  # 2 retries for connection errors
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            
            # Right-size pool based on global concurrency limiter
            # Pool should match actual concurrency limit, not exceed it
            from llm_concurrency import LLMConcurrencyLimiter
            limiter = LLMConcurrencyLimiter()
            pool_maxsize = limiter.global_limit  # Match semaphore limit exactly
            pool_connections = min(10, pool_maxsize // 3)  # 3-4 connections per pool
            
            adapter = HTTPAdapter(
                max_retries=retry_strategy,
                pool_connections=pool_connections,
                pool_maxsize=pool_maxsize,
                pool_block=True  # CRITICAL: Block and wait for connection, don't fail immediately
                # Note: pool_block_timeout is not supported by requests HTTPAdapter
                # Timeout is handled at the request level via timeout parameter
            )
            
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            # Initialize connection pool metrics
            try:
                from pipeline_metrics import connection_pool_metrics
                connection_pool_metrics.update_pool_size(0, pool_maxsize)
            except ImportError:
                pass  # Metrics not available
            
            _ollama_client = session
        except ImportError:
            raise ImportError("requests library required for Ollama. Install with: pip install requests")
    return _ollama_client


def get_openai_client():
    """Get OpenAI client"""
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            _openai_client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai library required. Install with: pip install openai")
    return _openai_client


def get_gemini_client():
    """Get Gemini client"""
    global _gemini_client
    if _gemini_client is None:
        try:
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not set")
            genai.configure(api_key=api_key)
            _gemini_client = genai
        except ImportError:
            raise ImportError("google-generativeai library required. Install with: pip install google-generativeai")
    return _gemini_client


def get_deepseek_client():
    """Get DeepSeek client (uses OpenAI-compatible API)"""
    global _deepseek_client
    if _deepseek_client is None:
        try:
            from openai import OpenAI
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY not set")
            base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            _deepseek_client = OpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            raise ImportError("openai library required for DeepSeek. Install with: pip install openai")
    return _deepseek_client


def call_ollama(prompt: str, system_prompt: Optional[str] = None, model: Optional[str] = None, max_retries: int = 3) -> str:
    """Call Ollama API with retry logic and adaptive timeout"""
    import requests
    import time
    import random
    
    # Acquire concurrency limiter before making request
    limiter = LLMConcurrencyLimiter()
    
    with limiter.acquire("ollama"):
        requests_lib = get_ollama_client()
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        # Default to phi3:mini - a fast, efficient model suitable for summarization
        model = model or os.getenv("OLLAMA_MODEL", "phi3:mini")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Estimate tokens for adaptive timeout calculation
        # Rough estimate: ~4 chars per token, add 20% for prompt overhead
        estimated_tokens = len(prompt) / 4 * 1.2
        # Base timeout: 2s per token + 60s overhead, min 180s, max 600s
        base_timeout = max(180, min(600, int(estimated_tokens * 2 + 60)))
        
        options = {
            "num_predict": 2000,  # Allow up to 2000 tokens for comprehensive summaries
            "num_thread": 0,  # Auto-detect optimal thread count (0 = auto)
            "temperature": TEMPERATURE.get("ollama", 0.2),  # Low temperature for deterministic output
            # GPU/Metal acceleration is automatic on macOS - no configuration needed
        }
        
        # Add stop sequences if available
        stop_sequences = STOP_SEQUENCES.get("ollama", [])
        if stop_sequences:
            options["stop"] = stop_sequences
        
        for attempt in range(max_retries):
            try:
                # Exponential backoff with jitter for retries
                if attempt > 0:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(wait_time)
                    print(f"  Retrying Ollama request (attempt {attempt + 1}/{max_retries}, waited {wait_time:.1f}s)...")
                
                # Timeout: (connect_timeout, read_timeout)
                # connect_timeout: Time to establish connection (including pool wait if pool_block=True)
                # read_timeout: Time to read response after connection established
                response = requests_lib.post(
                    f"{ollama_url}/api/chat",
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": False,
                        "options": options
                    },
                    timeout=(30, base_timeout)  # 30s connection timeout (includes pool wait), adaptive read timeout
                )
                response.raise_for_status()
                result = response.json()
                
                # Debug logging for empty responses
                if "message" not in result:
                    import json
                    print(f"[DEBUG call_ollama] Response missing 'message' key. Full response: {json.dumps(result, indent=2)[:1000]}")
                    raise Exception("Invalid response format from Ollama: missing 'message' key")
                
                if "content" not in result["message"]:
                    import json
                    print(f"[DEBUG call_ollama] Response missing 'content' key. Message keys: {list(result['message'].keys())}")
                    print(f"[DEBUG call_ollama] Full response: {json.dumps(result, indent=2)[:1000]}")
                    raise Exception("Invalid response format from Ollama: missing 'content' key")
                
                content = result["message"]["content"]
                if not content or len(content.strip()) == 0:
                    import json
                    print(f"[DEBUG call_ollama] Empty content in response. Full response: {json.dumps(result, indent=2)[:1000]}")
                    
                    # Check if there's a thinking field
                    thinking_content = result["message"].get("thinking", "")
                    
                    # If this is the last attempt and we have substantial thinking content, use it as fallback
                    if attempt == max_retries - 1 and thinking_content and len(thinking_content.strip()) > 100:
                        print(f"[WARNING call_ollama] Using 'thinking' field as fallback on final attempt (length: {len(thinking_content)})")
                        return thinking_content.strip()
                    
                    # Otherwise, retry (unless it's the last attempt)
                    if attempt < max_retries - 1:
                        print(f"[DEBUG call_ollama] Retrying due to empty content (attempt {attempt + 1}/{max_retries})...")
                        continue
                    else:
                        # Last attempt failed - provide detailed error
                        if thinking_content:
                            raise Exception(f"Empty content in Ollama response after {max_retries} attempts. Model returned only thinking field ({len(thinking_content)} chars). This may indicate the model needs different configuration or the prompt needs adjustment.")
                        else:
                            raise Exception(f"Empty content in Ollama response after {max_retries} attempts. No thinking field available.")
                
                return content.strip()
                
            except requests.exceptions.Timeout as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Ollama timeout: Request took too long after {max_retries} attempts (timeout={base_timeout}s)")
                # Retry on timeout with exponential backoff
                continue
            except requests.exceptions.ConnectionError as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Ollama connection error: Cannot connect to {ollama_url}. Is Ollama running?")
                # Retry on connection error
                continue
            except requests.exceptions.HTTPError as e:
                if e.response and e.response.status_code == 404:
                    error_detail = ""
                    try:
                        error_data = e.response.json()
                        if "error" in error_data:
                            error_detail = f" - {error_data['error']}"
                    except:
                        pass
                    raise Exception(f"Model '{model}' not found in Ollama{error_detail}. Available models: gpt-oss:20b, phi3:mini, deepseek-r1:8b. Set OLLAMA_MODEL environment variable to use a different model.")
                # Retry on rate limit (429) or server errors (5xx)
                if e.response and (e.response.status_code == 429 or e.response.status_code >= 500):
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        time.sleep(wait_time)
                        continue
                raise Exception(f"Ollama API error: {str(e)}")
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Ollama API error: {str(e)}")
                # Retry on other exceptions
                continue
        
        raise Exception("Max retries exceeded")


def call_openai(prompt: str, system_prompt: Optional[str] = None, model: Optional[str] = None) -> str:
    """Call OpenAI API"""
    limiter = LLMConcurrencyLimiter()
    
    with limiter.acquire("openai"):
        client = get_openai_client()
        model = model or os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            stop_sequences = STOP_SEQUENCES.get("openai", [])
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE.get("openai", 0.3),
                stop=stop_sequences if stop_sequences else None
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")


def call_gemini(prompt: str, system_prompt: Optional[str] = None, model: Optional[str] = None) -> str:
    """Call Gemini API"""
    limiter = LLMConcurrencyLimiter()
    
    with limiter.acquire("gemini"):
        genai = get_gemini_client()
        model_name = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        
        try:
            gen_model = genai.GenerativeModel(model_name)
            
            # Combine system prompt and user prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            generation_config = {
                "temperature": TEMPERATURE.get("gemini", 0.1),
            }
            stop_sequences = STOP_SEQUENCES.get("gemini", [])
            if stop_sequences:
                generation_config["stop_sequences"] = stop_sequences
            
            response = gen_model.generate_content(full_prompt, generation_config=generation_config)
            return response.text.strip()
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")


def call_deepseek(prompt: str, system_prompt: Optional[str] = None, model: Optional[str] = None) -> str:
    """Call DeepSeek API (OpenAI-compatible)"""
    limiter = LLMConcurrencyLimiter()
    
    with limiter.acquire("deepseek"):
        client = get_deepseek_client()
        model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            stop_sequences = STOP_SEQUENCES.get("deepseek", [])
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE.get("deepseek", 0.2),
                stop=stop_sequences if stop_sequences else None
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"DeepSeek API error: {str(e)}")


def call_llm(prompt: str, system_prompt: Optional[str] = None, provider: Optional[str] = None, model: Optional[str] = None) -> str:
    """
    Unified function to call any LLM provider.
    
    Args:
        prompt: User prompt
        system_prompt: Optional system prompt
        provider: Provider name ('ollama', 'openai', 'gemini', 'deepseek'). 
                  If None, uses OLLAMA by default.
        model: Optional model name (overrides env var)
    
    Returns:
        Generated text
    """
    provider = provider or os.getenv("API_PROVIDER", "ollama").lower()
    
    if provider == "ollama":
        return call_ollama(prompt, system_prompt, model)
    elif provider == "openai":
        return call_openai(prompt, system_prompt, model)
    elif provider == "gemini":
        return call_gemini(prompt, system_prompt, model)
    elif provider == "deepseek":
        return call_deepseek(prompt, system_prompt, model)
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: ollama, openai, gemini, deepseek")

