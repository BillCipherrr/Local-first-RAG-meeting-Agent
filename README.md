# Real-time Meeting RAG Agent
https://github.com/user-attachments/assets/994363ea-f382-4d4d-b8a0-ac14aa0f3c25

## Project Overview
This is a **local-first real-time meeting assistant** designed to transcribe speech and provide instant answers based on meeting context. It ensures data privacy by running entirely on your machine without relying on cloud APIs.

**Key Features:**
*   **Real-time Transcription:** Converts speech to text instantly using `faster-whisper`.
*   **AI Refinement & Minutes:** Uses LLM (Gemini) to correct transcripts based on agenda and generate structured meeting minutes.
*   **RAG (Retrieval-Augmented Generation):** Stores transcripts in a local `Qdrant` vector database, allowing you to ask questions about the meeting history.
*   **Knowledge Map:** Visualizes conversation topics in an interactive 3D space.
*   **Multi-language Support:** Supports Traditional Chinese, English, Japanese, and Korean.

## Setup

1.  **Create Conda Environment:**
    ```bash
    conda env create -f environment.yml
    conda activate meeting_rag_agent
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API Key:**
    Create a `.env` file in the root directory and add your Google Gemini API key:
    ```env
    GOOGLE_API_KEY=your_api_key_here
    ```
    *Note: Currently using Google Gemini 2.5 Flash for testing high-performance context handling. Future updates will add support for Local LLM APIs (e.g., Ollama) for a fully offline experience.*

## Running the Application

1.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

## Architecture

*   **STT:** Faster-Whisper (Local) running in a background thread.
*   **RAG:** Qdrant (Local) + SentenceTransformers.
*   **UI:** Streamlit with auto-refresh loop.
