[Refined v3] Project Ticket: Real-time Meeting RAG Agent (Local + Qdrant)

1. Summary
開發一個基於 Streamlit 的即時會議輔助系統。本版本將技術棧從 Cloud API 轉向 Local-First 架構，使用 Faster-Whisper 進行本地語音轉錄，並使用 Qdrant 進行向量儲存。
**新增功能 (v3)**: 引入「會後編輯模式」，允許使用者在停止錄音後，對轉錄結果進行檢視、修改或刪除，確認無誤後才寫入向量資料庫。

2. Technical Constraints (Updated)
Framework: Python 3.10+, Streamlit.

STT Engine: Faster-Whisper (Local Model).

Model Size: 使用 base 或 small 模型以確保即時性 (Latency < 2s)。

Compute: 必須檢查是否有 CUDA，若有則使用 GPU 加速，否則使用 CPU (int8 quantization)。

VAD: 必須實作 Voice Activity Detection (可以使用 webrtcvad 或 model 內建參數) 避免在靜音時產生幻覺文字。

Vector Database: Qdrant (Local Mode).

Initialization: client = QdrantClient(path="./qdrant_data")。

Collection Name: meeting_transcripts。

Concurrency: 依然維持 Producer-Consumer Pattern (STT 在背景 Thread 跑，Streamlit 透過 Queue 讀取)。

**UI Components**: 使用 `st.data_editor` 或類似元件實現互動式列表編輯。

3. Implementation Steps (Updated)
Step 1: Core Logic - Local STT (Faster-Whisper)
Task: 建立 stt_service.py。

Details:

載入 faster_whisper.WhisperModel。

實作一個 listen_and_transcribe() 函數，使用 pyaudio 錄製音訊 chunks。

關鍵邏輯：因為 Whisper 不是真正的 Streaming，你需要實作 "Audio Chunking" —— 每錄製 3-5 秒音訊（或偵測到停頓）就送進模型辨識一次。

Validation: 建立 test_stt_local.py。執行後對著麥克風說話，Terminal 需每隔幾秒印出辨識結果。確認 GPU/CPU 負載正常。

Step 2: Core Logic - RAG with Qdrant
Task: 建立 rag_service.py。

Details:

安裝 qdrant-client。

實作 add_transcript(text): 將文字轉 Embedding (可繼續用 Gemini 或改用 sentence-transformers 做全本地化) 後存入 Qdrant。

實作 query(text): 使用 Qdrant 的 search 方法找尋相似內容。

*Note*: 需支援批次寫入 (Batch Insert) 以優化會後儲存效能。

Validation: 建立 test_qdrant.py。存入 "Meeting started at 10 AM"，查詢 "When did it start?"，確認 Qdrant 回傳正確 Payload。

Step 3: Streamlit UI Integration (Updated)
Task: 建立/修改 app.py。

Details:

*   **Recording State**: 使用 threading 啟動 Transcriber，將結果放入 `queue.Queue` 並即時更新到 `st.session_state.transcript` (List of dicts: `{'id': 1, 'text': '...'}`).
*   **Review State**: 停止錄音後，進入編輯模式。
    *   隱藏 Chat Interface (或暫用唯讀)。
    *   顯示 `st.data_editor` 供使用者修改文字或勾選刪除。
*   **Finalize**: 使用者點擊 "Save to Knowledge Base" 後：
    *   過濾掉被標記刪除的句子。
    *   將最終文字列表傳送給 `rag_service` 進行 Embedding 與儲存。
    *   啟用 Chat Interface。

Validation: 啟動網頁，錄製一段話，停止後修改其中一句，確認儲存後的 Qdrant 搜尋結果反映的是修改後的內容。