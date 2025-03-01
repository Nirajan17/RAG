from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.memory import ConversationBufferMemory
from operator import itemgetter


MODEL = "llama3.2"
model = Ollama(model=MODEL)
embeddings = OllamaEmbeddings(model=MODEL)


memory = ConversationBufferMemory(
    memory_key="chat_history",  
    return_messages=True        
)

template = """
You are an AI assistant to the user. Provide response based on the context and the chat history.

Context: {context}
Chat History: {chat_history}
Question: {question}

"""
prompt = PromptTemplate.from_template(template)

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
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        
        vectorstore = DocArrayInMemorySearch.from_documents(
            pages, 
            embeddings
        )
        retriever = vectorstore.as_retriever()
        
        chain = (
            {
                "context": itemgetter("question") | retriever,
                "chat_history": lambda x: memory.load_memory_variables({})["chat_history"],
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

def ask_question(pdf_path, question):
    """
    Ask a question based on the content of the uploaded PDF and maintain chat history.
    
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
            
            memory.save_context({"question": question}, {"answer": answer})
            
            return answer
        except Exception as e:
            return f"Error answering question: {str(e)}"
    return "I couldn't answer that question"
