import os
import json
import google.generativeai as genai
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class LLMService:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM Service with Google Gemini.
        """
        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            # We don't raise error here to allow app to start, 
            # but methods will fail if called without key.
            print("Warning: GOOGLE_API_KEY not found in environment variables or .env file.")
            self.model = None
        else:
            genai.configure(api_key=api_key)
            # Using gemini-2.5-flash for speed and large context window
            self.model = genai.GenerativeModel('gemini-2.5-flash')

    def refine_transcript(self, agenda: str, raw_transcripts: List[Dict]) -> List[Dict]:
        """
        Refines the raw transcripts based on the agenda using Gemini.
        Returns a list of dicts with the same IDs but corrected text.
        """
        if not self.model:
            return raw_transcripts

        if not raw_transcripts:
            return []

        # Prepare the input for the model
        transcript_text = "\n".join([f"ID {t['id']}: {t['text']}" for t in raw_transcripts])
        
        prompt = f"""
        You are a professional meeting assistant. Your task is to refine the following raw meeting transcript based on the provided meeting agenda.
        
        Meeting Agenda:
        {agenda}
        
        Raw Transcript (ID: Text):
        {transcript_text}
        
        Instructions:
        1. Correct typos, grammatical errors, and ASR (Automatic Speech Recognition) misinterpretations.
        2. Ensure the context aligns with the agenda.
        3. Do NOT change the meaning of the speakers.
        4. IMPORTANT: You must return the result as a valid JSON list of objects.
        5. Each object must have exactly two fields: 'id' (integer, matching the input) and 'text' (string, the refined text).
        6. Do not output markdown code blocks (like ```json), just the raw JSON string.
        """
        
        try:
            response = self.model.generate_content(prompt)
            text_response = response.text.strip()
            
            # Clean up potential markdown formatting
            if text_response.startswith("```json"):
                text_response = text_response[7:]
            if text_response.startswith("```"):
                text_response = text_response[3:]
            if text_response.endswith("```"):
                text_response = text_response[:-3]
            
            refined_data = json.loads(text_response)
            
            # Validate structure
            if isinstance(refined_data, list):
                # Ensure we preserve the original structure/order if possible, 
                # or just return what the LLM gave if it matches our schema.
                return refined_data
            else:
                print("LLM returned invalid JSON structure (not a list).")
                return raw_transcripts
                
        except Exception as e:
            print(f"Error refining transcript: {e}")
            return raw_transcripts

    def generate_minutes(self, agenda: str, refined_transcripts: List[Dict]) -> str:
        """
        Generates structured meeting minutes in Markdown based on the transcript.
        """
        if not self.model:
            return "Error: Google API Key not configured."

        transcript_text = "\n".join([f"- {t['text']}" for t in refined_transcripts])
        
        prompt = f"""
        You are a professional minute-taker. Generate a structured meeting minutes document in Markdown format based on the provided agenda and transcript.
        
        Meeting Agenda:
        {agenda}
        
        Transcript:
        {transcript_text}
        
        The output should be a well-formatted Markdown document including:
        # Meeting Minutes
        ## Date: [Current Date]
        ## Executive Summary
        ## Key Discussion Points (aligned with Agenda)
        ## Action Items (Who, What, When)
        ## Decisions Made
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating minutes: {e}"

    def answer_question(self, context: str, question: str) -> str:
        """
        Answers a question based on the provided context using Gemini.
        """
        if not self.model:
            return "Error: Google API Key not configured."

        prompt = f"""
        You are a helpful meeting assistant. Answer the user's question based ONLY on the provided meeting context.
        If the answer is not in the context, say "I don't have enough information from the meeting to answer that."
        
        Meeting Context:
        {context}
        
        User Question:
        {question}
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error answering question: {e}"

