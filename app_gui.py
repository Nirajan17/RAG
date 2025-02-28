# app.py
import streamlit as st
import os
from app import ask_question
from datetime import datetime

# Streamlit app configuration
st.set_page_config(page_title="PDF Q&A", page_icon="📄")

# Title
st.title("PDF Question Answering")
st.write("Upload a PDF and ask questions. Scroll to see previous responses.")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Initialize session state for history and file path
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "temp_pdf_path" not in st.session_state:
    st.session_state.temp_pdf_path = None

# Handle file upload
if uploaded_file is not None:
    # Clean up previous file if it exists
    if st.session_state.temp_pdf_path and os.path.exists(st.session_state.temp_pdf_path):
        os.remove(st.session_state.temp_pdf_path)
    
    # Save new file temporarily
    temp_pdf_path = f"temp_{uploaded_file.name}"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.temp_pdf_path = temp_pdf_path
    st.success(f"Uploaded: {uploaded_file.name}")

    # Question input and submission
    question = st.text_input("Ask a question:")
    if st.button("Submit") and question:
        with st.spinner("Processing..."):
            answer = ask_question(temp_pdf_path, question)
            st.session_state.qa_history.append({
                "question": question,
                "answer": answer,
                "timestamp": datetime.now()
            })

# Display scrollable history
if st.session_state.qa_history:
    st.markdown("---")
    st.subheader("Previous Questions & Answers")
    for idx, entry in enumerate(st.session_state.qa_history):
        st.write(f"**Q{idx + 1} ({entry['timestamp'].strftime('%H:%M:%S')}):** {entry['question']}")
        st.write(f"**Answer:** {entry['answer']}")
        st.markdown("---")

# Clear button
if st.button("Clear All"):
    if st.session_state.temp_pdf_path and os.path.exists(st.session_state.temp_pdf_path):
        os.remove(st.session_state.temp_pdf_path)
    st.session_state.qa_history = []
    st.session_state.temp_pdf_path = None
    st.success("Cleared all data.")

# Footer
st.markdown("---")
st.write("Powered by LangChain & Ollama")

# Automatic cleanup on app close
def cleanup():
    if st.session_state.temp_pdf_path and os.path.exists(st.session_state.temp_pdf_path):
        os.remove(st.session_state.temp_pdf_path)

import atexit
atexit.register(cleanup)