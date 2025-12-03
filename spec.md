[Refined v4] Project Ticket: Real-time Meeting RAG Agent (LLM Enhanced)

1. Summary
開發一個基於 Streamlit 的即時會議輔助系統。本版本在 Local-First 架構基礎上，引入 LLM (Google Gemini) 進行智慧化處理。
**新增功能 (v4)**:
1.  **會前**: 支援輸入會議大綱 (Agenda) 作為 Context。
2.  **會後**: LLM 根據大綱自動修正 Whisper 逐字稿 (Refinement)，提升準確度。
3.  **輸出**: 自動生成結構化 Markdown 會議記錄，並將修正後的內容存入 Qdrant。

2. Technical Constraints (Updated v4)
Framework: Python 3.10+, Streamlit.

STT Engine: Faster-Whisper (Local Model).
*   Model Size: base/small (Latency < 2s).
*   Compute: CUDA if available, else CPU (int8).
*   VAD: Required to prevent hallucinations.

LLM Engine: Google Gemini API (Initial).
*   Purpose: Transcript Refinement & Minutes Generation.
*   Future-proof: Design `LLMService` to be swappable with Local LLM (Ollama).

Vector Database: Qdrant (Local Mode).
*   Path: `./qdrant_data` (SQLite backend).
*   Collection: `meeting_transcripts`.

Data Formats:
*   **Agenda**: Text/Markdown input.
*   **Minutes**: Markdown file output.
*   **RAG Storage**: Qdrant Vector Store.

Concurrency: Producer-Consumer Pattern (STT Thread -> Queue -> Streamlit).

3. Implementation Steps (Updated v4)
Step 1: Core Logic - Local STT (Faster-Whisper)
Task: 建立 stt_service.py (Existing).
*   維持現有邏輯：Audio Chunking -> Whisper -> Queue.

Step 2: Core Logic - RAG with Qdrant
Task: 建立 rag_service.py (Existing).
*   維持現有邏輯：Embeddings (Local) -> Qdrant Upsert/Search.

Step 3: Core Logic - LLM Service (New)
Task: 建立 `llm_service.py`.
*   **Setup**: 整合 `google-generativeai`.
*   **Refine**: 實作 `refine_transcript(agenda, raw_transcripts)` -> List[Dict].
    *   Prompt: "You are a meeting assistant. Given the agenda and raw transcript, correct typos and context errors."
*   **Minutes**: 實作 `generate_minutes(agenda, refined_transcripts)` -> Markdown String.
    *   Prompt: "Generate a structured meeting minutes in Markdown based on the transcript."

Step 4: Streamlit UI Integration (Updated)
Task: 修改 `app.py`.
*   **Sidebar**: 新增 `st.text_area` 輸入 Meeting Agenda.
*   **Post-Recording Workflow**:
    1.  Stop Recording.
    2.  **Auto-Refine**: 若有 Agenda，呼叫 `llm_service.refine_transcript` (顯示 Spinner).
    3.  **Review**: 顯示 `st.data_editor` (內容為 LLM 修正後的版本).
*   **Finalize**:
    1.  User clicks "Save & Generate Minutes".
    2.  Save to Qdrant (RAG).
    3.  Call `llm_service.generate_minutes`.
    4.  Display Minutes & Provide Download Button (`.md`).

Validation:
1.  輸入大綱 "Discuss Q3 Budget".
2.  錄音 "We need to cut costs by 10%".
3.  確認 LLM 修正後的文字無誤。
4.  確認生成的 Markdown 包含 "Action Item: Cut costs by 10%".