# Real-time Meeting RAG Agent

## Project Overview
This is a **local-first real-time meeting assistant** designed to transcribe speech and provide instant answers based on meeting context. It ensures data privacy by running entirely on your machine without relying on cloud APIs.

**Key Features:**
*   **Real-time Transcription:** Converts speech to text instantly using `faster-whisper`.
*   **RAG (Retrieval-Augmented Generation):** Stores transcripts in a local `Qdrant` vector database, allowing you to ask questions about the meeting history.
*   **Knowledge Map:** Visualizes conversation topics in an interactive 3D space.
*   **Multi-language Support:** Supports Traditional Chinese, English, Japanese, and Korean.

## Setup

1.  **Create Conda Environment:**
    ```bash
    conda env create -f environment.yml
    conda activate meeting_rag_agent
    ```

2.  **Install Dependencies (if not already installed):**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: `requirements.txt` contains the same dependencies as `environment.yml`)*

## Running the Application

1.  **Validation:**
    *   Test STT (Microphone required):
        ```bash
        python test_stt_local.py
        ```
    *   Test RAG (Qdrant):
        ```bash
        python test_qdrant.py
        ```

2.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

## Architecture

*   **STT:** Faster-Whisper (Local) running in a background thread.
*   **RAG:** Qdrant (Local) + SentenceTransformers.
*   **UI:** Streamlit with auto-refresh loop.
