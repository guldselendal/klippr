#!/usr/bin/env python3
"""
Debug script to test upload functionality and identify issues.
"""
import requests
import os
import sys

def test_backend_health():
    """Test if backend is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            print("✓ Backend is running")
            return True
        else:
            print(f"✗ Backend returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to backend. Is it running on http://localhost:8000?")
        return False
    except Exception as e:
        print(f"✗ Backend health check failed: {e}")
        return False

def test_ollama_health():
    """Test if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("✓ Ollama is running")
            models = response.json().get("models", [])
            if models:
                print(f"  Available models: {', '.join([m.get('name', 'unknown') for m in models])}")
            else:
                print("  Warning: No models found in Ollama")
            return True
        else:
            print(f"✗ Ollama returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to Ollama. Is it running on http://localhost:11434?")
        print("  Start Ollama with: ollama serve")
        return False
    except Exception as e:
        print(f"✗ Ollama health check failed: {e}")
        return False

def test_upload_endpoint():
    """Test upload endpoint with a small test file"""
    print("\nTesting upload endpoint...")
    
    # Create a minimal test file
    test_file_path = "/tmp/test_upload.txt"
    with open(test_file_path, "w") as f:
        f.write("Test content")
    
    try:
        with open(test_file_path, "rb") as f:
            files = {"file": ("test.txt", f, "text/plain")}
            response = requests.post(
                "http://localhost:8000/api/upload",
                files=files,
                timeout=5
            )
        
        if response.status_code == 400:
            error_detail = response.json().get("detail", "")
            if "Only EPUB and PDF" in error_detail:
                print("✓ Upload endpoint is working (correctly rejected non-PDF/EPUB)")
                return True
            else:
                print(f"✗ Upload endpoint error: {error_detail}")
                return False
        else:
            print(f"✗ Unexpected response: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            return False
    except requests.exceptions.Timeout:
        print("✗ Upload endpoint timed out")
        return False
    except Exception as e:
        print(f"✗ Upload endpoint test failed: {e}")
        return False
    finally:
        if os.path.exists(test_file_path):
            os.remove(test_file_path)

def check_upload_directory():
    """Check if upload directory exists and is writable"""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    upload_dir = os.path.join(script_dir, "uploads")
    
    if not os.path.exists(upload_dir):
        print(f"✗ Upload directory '{upload_dir}' does not exist")
        print(f"  Creating directory...")
        try:
            os.makedirs(upload_dir, exist_ok=True)
            print(f"  ✓ Created upload directory")
        except Exception as e:
            print(f"  ✗ Failed to create directory: {e}")
            return False
    
    if not os.access(upload_dir, os.W_OK):
        print(f"✗ Upload directory '{upload_dir}' is not writable")
        return False
    
    print(f"✓ Upload directory '{upload_dir}' exists and is writable")
    return True

def main():
    print("=" * 60)
    print("Upload Debugging Tool")
    print("=" * 60)
    
    results = []
    
    print("\n1. Checking backend health...")
    results.append(("Backend", test_backend_health()))
    
    print("\n2. Checking Ollama health...")
    results.append(("Ollama", test_ollama_health()))
    
    print("\n3. Checking upload directory...")
    results.append(("Upload Directory", check_upload_directory()))
    
    print("\n4. Testing upload endpoint...")
    results.append(("Upload Endpoint", test_upload_endpoint()))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_ok = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
        if not result:
            all_ok = False
    
    if all_ok:
        print("\n✓ All checks passed! Upload should work.")
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        print("\nNext steps:")
        print("1. Check browser console for specific error messages")
        print("2. Check backend logs for detailed error information")
        print("3. Try uploading a small test file (< 1 MB)")
        print("4. Verify file is valid EPUB or PDF format")

if __name__ == "__main__":
    main()

