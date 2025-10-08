import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings()
faiss_path = "FAISS_db/RAG"

if os.path.exists(faiss_path):
    vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
else:
    loader = PyPDFLoader("Schiller_Mary_Stuart.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    texts = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(faiss_path)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4"),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

query = "Who did the translation of this book?"
response = qa_chain.invoke(query)
print(response)