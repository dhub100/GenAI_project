import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama


# from dotenv import load_dotenv

## Install Ollama locally:
# https://ollama.com/download

## Pull the model (bash)
# ollama pull llama3.1:8b

## Install libraries
# pip install langchain langchain-community sentence-transformers faiss-cpu python-dotenv pypdf langchain-ollama

## Start the server (bash)
# ollama serve



# Load environment variables (not needed anymore)
# load_dotenv()

# No API key needed for Ollama
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Use free Hugging Face embeddings instead of OpenAIEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

faiss_path = "FAISS_db_v2/RAG"

if os.path.exists(faiss_path):
    vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
else:
    loader = PyPDFLoader("Schiller_Mary_Stuart.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    texts = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(faiss_path)

# Use local Llama 3.1 model via Ollama
llm = ChatOllama(model="llama3.1:8b", temperature=0.1)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

query = "What is the relation between Mary and Elisabeth?"
response = qa_chain.invoke(query)
print(response)
