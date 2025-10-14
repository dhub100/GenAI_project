import os
from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from langchain.schema import Document
# from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

## Install Ollama locally:
# https://ollama.com/download

## Pull the model (bash)
# ollama pull llama3.1:8b

## Start the server (bash)
# ollama serve

### Semantic chunking function ###
def semantic_chunk_text(text, target_length=800, overlap_sentences=2):
    """
    Split text into semantic chunks based on sentences.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_chunk.append(sentence)
        if len(" ".join(current_chunk)) > target_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap_sentences:]

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


# Use free Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

faiss_path = "FAISS_db_Orwell/RAG"

if os.path.exists(faiss_path):
    vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
else:
    loader = PyPDFLoader("George_Orwell_1984.pdf")
    documents = loader.load()
    # --- Combine all pages into a single string ---
    full_text = " ".join([doc.page_content for doc in documents])
    # --- Apply semantic chunking ---
    semantic_chunks = semantic_chunk_text(full_text, target_length=800, overlap_sentences=2)
    # --- Convert chunks to LangChain Document objects ---
    texts = [Document(page_content=chunk) for chunk in semantic_chunks]
    # --- Create FAISS vectorstore ---
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(faiss_path)

# Use local Llama 3.1 model via Ollama
llm = ChatOllama(model="llama3.1:8b", temperature=0.1)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

query = "war is peace. freedom is slavery. What is ignorance? short answer."

print("💭 Thinking... (retrieving context and generating answer, this may take a while)")
response = qa_chain.invoke(query)
print("✅ Done!\n")
print(response)
