from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser  # Use the general OutputParser
from langchain_ollama import OllamaLLM
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
LANGSMITH_TRACING = True
LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
LANGSMITH_API_KEY = " "
LANGSMITH_PROJECT = "pr-bumpy-colon-8"

# Initialize OllamaLLM
ollama_llm = OllamaLLM(base_url="http://localhost:11434", model="llama3.2")

# Define prompt template
prompt = ChatPromptTemplate.from_messages([ 
    ("system", "You are a very helpful assistant. Please respond to the user queries."),
    ("user", "Question: {question}")
])

# Streamlit UI
st.title("LangChain Demi using Ollama 3.2")
input_txt = st.text_input("Search the topic you want to:")

# Process and invoke with OutputParser
outputParser = StrOutputParser()  # General output parser
chain = prompt | ollama_llm | outputParser

if input_txt:
    st.write(chain.invoke({"question": input_txt}))



# Data Loading and Chunking

from langchain.document_loaders import TextLoader, CSVLoader, PDFMinerLoader # Example loaders
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load data
loader = TextLoader("data.txt")  # Or CSVLoader, PDFMinerLoader, etc.
documents = loader.load()

# Chunk data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) # Adjust chunk size
chunks = text_splitter.split_documents(documents)


# Now use 'chunks' to create your vector database as before.
