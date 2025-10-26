import os

from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
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

faiss_path = "../RAG_Database/FAISS_db_Orwell/RAG"

if os.path.exists(faiss_path):
    vectorstore = FAISS.load_local(
        faiss_path, embeddings, allow_dangerous_deserialization=True
    )
else:
    loader = PyPDFLoader("../RAG_Database/George_Orwell_1984.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    texts = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(faiss_path)

# Use local Llama 3.1 model via Ollama
llm = ChatOllama(model="llama3.1:8b", temperature=0.1)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
)


def answer_query(query: str):
    """
    Runs the RAG pipeline on a given query and returns the model's answer.
    """
    print(
        "💭 Thinking... (retrieving context and generating answer, this may take a while)"
    )
    response = qa_chain.invoke(query)
    print("✅ Done!\n")
    if isinstance(response, dict) and "result" in response:
        final_answer = response["result"]
    else:
        final_answer = response
    return final_answer


# Run only when executed directly (not when imported)
if __name__ == "__main__":
    query = "What is the Junior Anti-Sex League Orwell is writing about?"
    answer = answer_query(query)
    print(answer)
