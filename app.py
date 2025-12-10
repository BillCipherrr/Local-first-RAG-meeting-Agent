import streamlit as st
import threading
import queue
import time
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
import numpy as np
from stt_service import STTService
from rag_service import RAGService
from llm_service import LLMService

# Initialize Services (Singleton pattern using st.cache_resource)
@st.cache_resource
def get_stt_service():
    # Using "small" model for better balance of speed and accuracy
    return STTService(model_size="medium")

@st.cache_resource
def get_rag_service():
    return RAGService()

@st.cache_resource
def get_llm_service():
    return LLMService()

stt_service = get_stt_service()
rag_service = get_rag_service()
llm_service = get_llm_service()

st.title("Real-time Meeting RAG Agent")

# Session State Initialization
if "transcripts" not in st.session_state:
    # Structure: [{"id": 0, "text": "...", "do_save": True}, ...]
    st.session_state.transcripts = []
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agenda" not in st.session_state:
    st.session_state.agenda = ""
if "minutes" not in st.session_state:
    st.session_state.minutes = ""
if "is_refined" not in st.session_state:
    st.session_state.is_refined = False

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    
    # Agenda Input
    st.subheader("Meeting Agenda")
    uploaded_file = st.file_uploader("Upload Agenda (.txt, .md)", type=["txt", "md"])
    if uploaded_file is not None:
        stringio = uploaded_file.getvalue().decode("utf-8")
        st.session_state.agenda = stringio
    
    agenda_input = st.text_area("Or paste agenda here:", value=st.session_state.agenda, height=150)
    st.session_state.agenda = agenda_input

    # Language Selection
    LANGUAGES = {
        "Traditional Chinese (繁體中文)": "zh",
        "English": "en",
        "Japanese": "ja",
        "Korean": "ko"
    }
    selected_lang_name = st.selectbox("Select Language", list(LANGUAGES.keys()), index=0)
    selected_lang_code = LANGUAGES[selected_lang_name]
    
    # Update STT Service language
    if stt_service.language != selected_lang_code:
        stt_service.set_language(selected_lang_code)

    if st.button("Start Recording"):
        if not st.session_state.is_recording:
            stt_service.start_recording()
            st.session_state.is_recording = True
            st.success("Recording started!")
            st.rerun()

    if st.button("Stop Recording"):
        if st.session_state.is_recording:
            stt_service.stop_recording()
            st.session_state.is_recording = False
            
            # Auto-Refine if Agenda exists
            if st.session_state.agenda and not st.session_state.is_refined:
                with st.spinner("AI is refining transcripts based on agenda..."):
                    refined = llm_service.refine_transcript(st.session_state.agenda, st.session_state.transcripts)
                    # Update transcripts
                    id_map = {r['id']: r['text'] for r in refined}
                    for t in st.session_state.transcripts:
                        if t['id'] in id_map:
                            t['text'] = id_map[t['id']]
                    st.session_state.is_refined = True
                    st.success("Transcripts refined by AI!")

            st.warning("Recording stopped! Please review transcripts.")
            st.rerun()

# Main Layout
tab1, tab2 = st.tabs(["Live Meeting", "Knowledge Map"])

with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Transcript")
        
        # Process new transcripts from Queue
        while not stt_service.transcript_queue.empty():
            text = stt_service.transcript_queue.get()
            # Append new transcript as a dict for the data editor
            new_id = len(st.session_state.transcripts)
            st.session_state.transcripts.append({"id": new_id, "text": text, "do_save": True})
            
        if st.session_state.is_recording:
            # Read-only view during recording
            transcript_container = st.container()
            with transcript_container:
                if not st.session_state.transcripts:
                    st.info("Listening...")
                
                # Only show the last 10 transcripts to avoid performance issues and WebSocket crashes
                recent_transcripts = st.session_state.transcripts[-10:]
                if len(st.session_state.transcripts) > 10:
                    st.text(f"... (Previous {len(st.session_state.transcripts) - 10} lines hidden) ...")
                
                for t in recent_transcripts:
                    # Handle legacy string format if any, though we initialized as list of dicts
                    text_content = t["text"] if isinstance(t, dict) else t
                    st.write(f"- {text_content}")
            
            # Auto-refresh mechanism
            time.sleep(1)
            st.rerun()
            
        else:
            # Review Mode: Editable Dataframe
            if st.session_state.transcripts:
                st.info("Review Mode: Edit text or uncheck 'do_save' to exclude sentences.")
                
                # Data Editor
                edited_df = st.data_editor(
                    st.session_state.transcripts,
                    column_config={
                        "text": st.column_config.TextColumn("Transcript", width="large"),
                        "do_save": st.column_config.CheckboxColumn("Save?", help="Uncheck to delete"),
                    },
                    use_container_width=True,
                    num_rows="dynamic"
                )
                
                # Save Button
                if st.button("Save & Generate Minutes", type="primary"):
                    # Filter transcripts
                    final_transcripts = [
                        row for row in edited_df 
                        if row.get("do_save", False) and row["text"].strip()
                    ]
                    final_texts = [t["text"] for t in final_transcripts]
                    
                    if final_texts:
                        with st.spinner("Saving to Vector Database & Generating Minutes..."):
                            # 1. Save to RAG
                            rag_service.batch_add_transcripts(final_texts)
                            
                            # 2. Generate Minutes
                            if st.session_state.agenda:
                                minutes = llm_service.generate_minutes(st.session_state.agenda, final_transcripts)
                                st.session_state.minutes = minutes
                            else:
                                st.session_state.minutes = "No agenda provided. Minutes generation skipped."

                        st.success(f"Saved {len(final_texts)} sentences to Knowledge Base!")
                        # Clear transcripts after saving
                        st.session_state.transcripts = []
                        st.session_state.is_refined = False # Reset for next time
                        st.rerun()
                    else:
                        st.warning("No transcripts selected to save.")
            else:
                st.info("No transcripts to review. Start recording to begin.")
                
        # Display Minutes if available
        if st.session_state.minutes:
            st.divider()
            st.subheader("Meeting Minutes")
            st.markdown(st.session_state.minutes)
            st.download_button("Download Minutes", st.session_state.minutes, file_name="meeting_minutes.md")

    with col2:
        st.subheader("AI Assistant")
        
        # Chat Interface
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about the meeting..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                # RAG Query
                context_results = rag_service.query(prompt)
                context_text = "\n".join(context_results)
                
                # Use Gemini to answer
                with st.spinner("Thinking..."):
                    response = llm_service.answer_question(context_text, prompt)
            
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

with tab2:
    st.subheader("Semantic Knowledge Map")
    if st.button("Refresh Map"):
        # Increased limit to visualize more history
        points = rag_service.get_all_vectors(limit=1000)
        if len(points) > 2:
            vectors = [p.vector for p in points]
            texts = [p.payload.get("text", "") for p in points]
            
            # Reduce dimensions to 3D using PCA for better visualization
            pca = PCA(n_components=3)
            vectors_3d = pca.fit_transform(vectors)
            
            df = pd.DataFrame(vectors_3d, columns=["x", "y", "z"])
            df["text"] = texts
            
            # Use 3D Scatter plot
            fig = px.scatter_3d(
                df, 
                x="x", y="y", z="z", 
                hover_data=["text"], 
                title="3D Transcript Clusters",
                color="z", # Color by Z-axis depth
                opacity=0.7
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data points to visualize yet. Record more audio!")
