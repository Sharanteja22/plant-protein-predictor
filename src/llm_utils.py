import os
import streamlit as st
from google import genai
from dotenv import load_dotenv

load_dotenv()

# For local development
api_key = os.getenv("GEMINI_API_KEY")

# For Streamlit Cloud deployment
if not api_key and "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]

def generate_explanation(go_terms):

    if not api_key:
        return "Gemini API key not found."

    client = genai.Client(api_key=api_key)

    prompt = f"""
    A plant protein was predicted to have the following Molecular Function GO terms:

    {go_terms}

    Explain in simple biological language what this protein likely does.
    Keep it concise and suitable for biology students.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text.strip()
