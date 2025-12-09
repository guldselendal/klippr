# Klippr Frontend

A modern React frontend for the Klippr incremental reading application.

## Features

- **Document Library**: View and manage all uploaded EPUB and PDF documents
- **Incremental Reading**: Read chapters one at a time with easy navigation
- **Chapter Connections**: Discover relationships between chapters across documents
- **Batch Operations**: Upload and delete multiple documents at once
- **Modern UI**: Clean, responsive design with smooth interactions

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`.

## Configuration

The frontend is configured to connect to the backend API at `http://localhost:8000` by default. This is handled via the Vite proxy configuration in `vite.config.ts`.

If you need to change the API URL, you can:
- Set the `VITE_API_URL` environment variable
- Or modify the proxy settings in `vite.config.ts`

## Project Structure

```
frontend/
├── src/
│   ├── api/           # API client and types
│   ├── components/    # React components
│   │   ├── Library.tsx      # Document library view
│   │   ├── Reader.tsx       # Chapter reading view
│   │   ├── Connections.tsx  # Connections visualization
│   │   └── Navbar.tsx       # Navigation bar
│   ├── App.tsx        # Main app component with routing
│   └── main.tsx       # Entry point
├── package.json
└── vite.config.ts     # Vite configuration
```

## Usage

1. **Upload Documents**: Click "Upload Document" in the Library view to upload EPUB or PDF files
2. **Read Chapters**: Click on a document or use the "Read" button to start reading
3. **Navigate**: Use Previous/Next buttons or the chapters sidebar to navigate
4. **View Connections**: Visit the Connections page to see relationships between chapters
5. **Manage Library**: Select multiple documents and delete them in batch

## Building for Production

```bash
npm run build
```

The built files will be in the `dist` directory.

