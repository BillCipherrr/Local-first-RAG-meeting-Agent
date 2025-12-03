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

# Initialize Services (Singleton pattern using st.cache_resource)
@st.cache_resource
def get_stt_service():
    # Using "small" model for better balance of speed and accuracy
    return STTService(model_size="medium")

@st.cache_resource
def get_rag_service():
    return RAGService()

stt_service = get_stt_service()
rag_service = get_rag_service()

st.title("Real-time Meeting RAG Agent")

# Session State Initialization
if "transcripts" not in st.session_state:
    st.session_state.transcripts = []
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    
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

    if st.button("Stop Recording"):
        if st.session_state.is_recording:
            stt_service.stop_recording()
            st.session_state.is_recording = False
            st.warning("Recording stopped!")

# Main Layout
tab1, tab2 = st.tabs(["Live Meeting", "Knowledge Map"])

with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Transcript")
        transcript_container = st.container()
        
        # Process new transcripts from Queue
        while not stt_service.transcript_queue.empty():
            text = stt_service.transcript_queue.get()
            st.session_state.transcripts.append(text)
            # Add to RAG immediately
            rag_service.add_transcript(text)
            
        with transcript_container:
            for t in st.session_state.transcripts:
                st.write(f"- {t}")

        # Auto-refresh mechanism
        if st.session_state.is_recording:
            time.sleep(1)
            st.rerun()

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
                
                # Simple response generation (Mocking an LLM here as per spec focus on RAG structure)
                # In a real scenario, you'd pass 'context_text' to an LLM.
                response = f"Based on the meeting context:\n\n{context_text}"
            
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

with tab2:
    st.subheader("Semantic Knowledge Map")
    if st.button("Refresh Map"):
        points = rag_service.get_all_vectors(limit=200)
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
