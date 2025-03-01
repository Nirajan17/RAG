import streamlit as st
import os
from app import ask_question  


# Streamlit app
def main():
    st.title("PDF Chatbot with Memory")
    st.write("Upload a PDF and ask questions about its content. The chatbot remembers the conversation!")

    # Sidebar for PDF upload
    with st.sidebar:
        st.header("Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            pdf_path = f"temp_{uploaded_file.name}"
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("PDF uploaded successfully!")
        else:
            pdf_path = None

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input for new question
    if pdf_path:
        question = st.chat_input("Ask a question about the PDF...")
        if question:
            # Display user question
            with st.chat_message("user"):
                st.markdown(question)
            st.session_state.messages.append({"role": "user", "content": question})

            # Get and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = ask_question(pdf_path, question)
                    st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.warning("Please upload a PDF to start chatting.")

    # Clean up temporary file if it exists
    if pdf_path and os.path.exists(pdf_path):
        os.remove(pdf_path)

if __name__ == "__main__":
    main()