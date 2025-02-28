from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import DocArrayInMemorySearch
from operator import itemgetter

# Model configuration
MODEL = "llama3.2"
model = Ollama(model=MODEL)
embeddings = OllamaEmbeddings(model=MODEL)

# Define the prompt template
template = """
Answer the questions based on the context below. If you cannot answer the question given, just reply I don't know

Context: {context}
Question: {question}
"""
prompt = PromptTemplate.from_template(template)

# Parser for output
parser = StrOutputParser()

def process_pdf_and_create_chain(pdf_path):
    """
    Process a PDF file from the given path and create a question-answering chain.
    
    Args:
        pdf_path (str): Path to the PDF file uploaded by the user
        
    Returns:
        Chain object for question answering
    """
    try:
        # Load and split the PDF
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        
        # Create vector store from the PDF pages
        vectorstore = DocArrayInMemorySearch.from_documents(
            pages, 
            embeddings
        )
        retriever = vectorstore.as_retriever()
        
        # Create the question-answering chain
        chain = (
            {
                "context": itemgetter("question") | retriever,
                "question": itemgetter("question")
            }
            | prompt
            | model
            | parser
        )
        
        return chain
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return None

# Example usage function
def ask_question(pdf_path, question):
    """
    Ask a question based on the content of the uploaded PDF.
    
    Args:
        pdf_path (str): Path to the PDF file
        question (str): Question to ask about the PDF content
        
    Returns:
        str: Answer to the question or error message
    """
    chain = process_pdf_and_create_chain(pdf_path)
    if chain:
        try:
            answer = chain.invoke({"question": question})
            return answer
        except Exception as e:
            return f"Error answering question: {str(e)}"
    return "I don't know"
