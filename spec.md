[Refined v2] Project Ticket: Real-time Meeting RAG Agent (Local + Qdrant)

1. Summary
開發一個基於 Streamlit 的即時會議輔助系統。本版本將技術棧從 Cloud API 轉向 Local-First 架構，使用 Faster-Whisper 進行本地語音轉錄，並使用 Qdrant 進行向量儲存。

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

Validation: 建立 test_qdrant.py。存入 "Meeting started at 10 AM"，查詢 "When did it start?"，確認 Qdrant 回傳正確 Payload。

Step 3: Streamlit UI Integration
Task: 建立 app.py。

Details:

使用 threading 啟動 Step 1 的 Transcriber，將結果放入 queue.Queue。

利用 st.session_state 儲存 transcript list。

使用 streamit.empty() 或 st.rerun() 讓 UI 定期刷新讀取 Queue 中的新文字。

整合 Chat Interface 呼叫 Step 2 的 rag_service。

Validation: 啟動網頁，確認長時間錄音（>1分鐘）不會造成 Memory Leak 或 UI 卡死。