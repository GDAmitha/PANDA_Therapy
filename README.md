# PANDA Therapy

PANDA Therapy is an AI-powered therapy dashboard that allows users to chat with an AI therapist, upload therapy transcripts (text or audio), and view insights.

## Requirements

- Python 3.9+ 
- Node.js 16+
- OpenAI API key (for transcript analysis and AI responses)
- Pinecone API key (for knowledge storage)

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/PANDA_Therapy.git
cd PANDA_Therapy
```

### 2. Backend Setup

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file in the root directory with your API keys:

```
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

> **Note**: If you encounter Pinecone compatibility issues, the system will automatically fall back to local vector storage. For development purposes, this is fine and no action is required.

### 3. Frontend Setup

```bash
cd frontend
npm install
```

### 4. Start the Application

In one terminal, start the backend:
```bash
cd backend
python main.py
```

In another terminal, start the frontend:
```bash
cd frontend
npm run dev
```

The application will be available at [http://localhost:3000](http://localhost:3000)

## Using the Application

1. **Chat with AI Therapist**: Navigate to the chat page to start a conversation with the AI therapist.

2. **Upload Transcripts**: 
   - Go to the transcripts page
   - Drag and drop or browse for text files (.txt, .md) or audio files (.wav, .mp3)
   - The system will process the files, transcribe audio if needed, and extract insights

3. **View Insights**: After uploading transcripts, you can view the extracted insights on the insights page.

## Troubleshooting

- **Authentication Issues**: The system automatically creates a default user for local development.
- **Transcript Processing Errors**: Make sure your OpenAI API key has sufficient credits.
- **Pinecone Connection Issues**: Verify your Pinecone API key is correct in the .env file. If problems persist, the system will fall back to local storage.
- **Pinecone API Error**: The wrapper supports multiple versions of the Pinecone client. If you encounter errors, try installing a specific version: `pip install pinecone-client==2.2.1`

## Advanced Setup

For advanced setup options or to configure the system for production use, please refer to the developer documentation.