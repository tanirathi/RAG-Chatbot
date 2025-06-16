# import os
# import streamlit as st
# import pandas as pd
# from langchain.document_loaders import CSVLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import SentenceTransformerEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from langchain_ollama import OllamaLLM
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # FAISS index path
# FAISS_INDEX_PATH = "faiss_index"

# # 1️⃣ **Load & Process Data Efficiently**
# @st.cache_resource
# def create_vector_d r"C\Users\Hp\Downloads\RAG\dialysis_data.csv"
#     """Loads CSV, formats data into structured text, and creates FAISS vectorstore."""
    
#     df = pd.read_csv(data_path)

#     # ✅ Convert dataset into structured text (AI understands this better)
#     records = df.to_dict(orient="records")
#     documents = [
#         f"Patient {rec['Patient_ID']} ({rec['Sex']}, {rec['Age']} years old)\n"
#         f"- Date: {rec['Datetime']}\n"
#         f"- Hypertension: {'Yes' if rec['Is_Hypertensive'] else 'No'}\n"
#         f"- Diabetes: {'Yes' if rec['Is_Diabetic'] else 'No'}\n"
#         f"- Dialyzer: {rec['Dialyzer']}, Technique: {rec['Type_of_Technique']}\n"
#         f"- Body Temperature: {rec['Body_Temperature']}°C, Heart Rate: {rec['Heart_Rate']} bpm\n"
#         f"- Blood Pressure: {rec['Systolic_BP']}/{rec['Diastolic_BP']} mmHg\n"
#         f"- Urea Clearance: {rec['Urea_Clearance']}, Volume Changes: {rec['Volume_Changes']}\n"
#         "------------------------------------------"
#         for rec in records
#     ]
    
#     # ✅ Ensure text is split properly for FAISS
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     chunks = text_splitter.create_documents(documents)

#     # ✅ Use Efficient Embedding Model
#     embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

#     # ✅ Load or Create FAISS Index
#     if os.path.exists(FAISS_INDEX_PATH):  
#         db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
#     else:
#         db = FAISS.from_documents(chunks, embeddings)
#         db.save_local(FAISS_INDEX_PATH)  
    
#     return db

# # 2️⃣ **Setup LLM & Prompt for Retrieval-Based QA**
# ollama_llm = OllamaLLM(base_url="http://localhost:11434", model="moondream")

# # ✅ **Improved Prompt for Better AI Answers**
# PROMPT_TEMPLATE = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""You are an AI trained to analyze dialysis patient records.
    
#     Given the following patient data:
#     {context}
    
#     Extract the precise answer from the dataset.
    
#     Question: {question}
    
#     If the answer is not found, respond with: "I could not find that information in the dataset."
#     """
# )

# # 3️⃣ **Streamlit UI: Chatbot**
# st.set_page_config(page_title="Dialysis Data Chatbot", page_icon="🤖", layout="wide")
# st.title("💬 Dialysis Data Chatbot")

# st.sidebar.title("⚙️ Upload Data")
# uploaded_file = st.sidebar.file_uploader("📂 Upload CSV File", type="csv")

# if uploaded_file:
#     db = create_vector_db(uploaded_file.name)
#     retriever = db.as_retriever()

#     # 🔹 **Chat Interface**
#     st.subheader("💡 Chat with the Data")
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     input_txt = st.text_input("🧐 Ask something about the data...")

#     if st.button("🔍 Get Answer"):
#         if input_txt:
#             with st.spinner("🔎 Searching..."):
#                 retrieved_docs = retriever.get_relevant_documents(input_txt)

#                 if not retrieved_docs:
#                     response = "⚠️ I could not find that information in the dataset."
#                 else:
#                     # ✅ **Pass structured context to the AI**
#                     context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    
#                     # ✅ **Use Improved Prompt to Ensure Correct Answer Extraction**
#                     qa_chain = RetrievalQA.from_chain_type(
#                         llm=ollama_llm,
#                         retriever=retriever,
#                         chain_type="stuff",
#                         return_source_documents=True,
#                         chain_type_kwargs={"prompt": PROMPT_TEMPLATE}
#                     )
#                     result = qa_chain({"query": input_txt, "context": context_text})
#                     response = result['result']

#                 # 🔹 **Update Chat History**
#                 st.session_state.chat_history.append(("User", input_txt))
#                 st.session_state.chat_history.append(("AI", response))

#                 # 🔹 **Display Chat**
#                 for sender, msg in st.session_state.chat_history:
#                     if sender == "User":
#                         st.markdown(f"**🧑‍💻 {sender}:** {msg}")
#                     else:
#                         st.markdown(f"**🤖 {sender}:** {msg}")

# else:
#     st.info("📂 Please upload a CSV file to start.")

import os
import streamlit as st
import pandas as pd
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# FAISS index path
FAISS_INDEX_PATH = "faiss_index"

# Set CSV file path directly
CSV_FILE_PATH = r"C:\Users\Hp\Downloads\RAG\dialysis_data.csv"

# 1️⃣ **Load & Process Data Efficiently**
@st.cache_resource
def create_vector_db(data_path):  
    """Loads CSV, formats data into structured text, and creates FAISS vectorstore."""

    # ✅ Load data directly from file path
    df = pd.read_csv(data_path)

    # ✅ Convert dataset into structured text
    records = df.to_dict(orient="records")
    documents = [
        f"Patient {rec['Patient_ID']} ({rec['Sex']}, {rec['Age']} years old)\n"
        f"- Date: {rec['Datetime']}\n"
        f"- Hypertension: {'Yes' if rec['Is_Hypertensive'] else 'No'}\n"
        f"- Diabetes: {'Yes' if rec['Is_Diabetic'] else 'No'}\n"
        f"- Dialyzer: {rec['Dialyzer']}, Technique: {rec['Type_of_Technique']}\n"
        f"- Body Temperature: {rec.get('Body_Temperature', 'N/A')}°C, Heart Rate: {rec.get('Heart_Rate', 'N/A')} bpm\n"
        f"- Blood Pressure: {rec.get('Systolic_BP', 'N/A')}/{rec.get('Diastolic_BP', 'N/A')} mmHg\n"
        f"- Urea Clearance: {rec['Urea_Clearance']}, Volume Changes: {rec['Volume_Changes']}\n"
        "------------------------------------------"
        for rec in records
    ]

    # ✅ Text Splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.create_documents(documents)

    # ✅ Embedding Model
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # ✅ FAISS Vector DB
    if os.path.exists(FAISS_INDEX_PATH):  
        db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(FAISS_INDEX_PATH)

    return db

# 2️⃣ **Setup LLM & Prompt**
ollama_llm = OllamaLLM(base_url="http://localhost:11434", model="moondream")

PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an AI trained to analyze dialysis patient records.
    
    Given the following patient data:
    {context}
    
    Extract the precise answer from the dataset.
    
    Question: {question}
    
    If the answer is not found, respond with: "I could not find that information in the dataset."
    """
)

# 3️⃣ **Streamlit UI**
st.set_page_config(page_title="Dialysis Data Chatbot", page_icon="🤖", layout="wide")
st.title("💬 Dialysis Data Chatbot")

# ✅ Load data directly without upload
st.sidebar.success("Using pre-loaded dataset from:")
st.sidebar.code(CSV_FILE_PATH)

# 🔹 Create vector DB
with st.spinner("🔄 Indexing the dataset..."):
    db = create_vector_db(CSV_FILE_PATH)
    retriever = db.as_retriever()

# 🔹 Chat Interface
st.subheader("💡 Chat with the Data")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

input_txt = st.text_input("🧐 Ask something about the data...")

if st.button("🔍 Get Answer"):
    if input_txt:
        with st.spinner("🔎 Searching..."):
            retrieved_docs = retriever.get_relevant_documents(input_txt)

            if not retrieved_docs:
                response = "⚠️ I could not find that information in the dataset."
            else:
                context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
                qa_chain = RetrievalQA.from_chain_type(
                    llm=ollama_llm,
                    retriever=retriever,
                    chain_type="stuff",
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": PROMPT_TEMPLATE}
                )
                result = qa_chain({"query": input_txt, "context": context_text})
                response = result['result']

        # 🔹 Update & Display Chat
        st.session_state.chat_history.append(("User", input_txt))
        st.session_state.chat_history.append(("AI", response))

        for sender, msg in st.session_state.chat_history:
            if sender == "User":
                st.markdown(f"**🧑‍💻 {sender}:** {msg}")
            else:
                st.markdown(f"**🤖 {sender}:** {msg}")
