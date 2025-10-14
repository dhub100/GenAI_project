import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama


#from dotenv import load_dotenv

## Install Ollama locally:
# https://ollama.com/download

## Pull the model (bash)
#ollama pull llama3.1:8b

## Install libraries
# pip install langchain langchain-community sentence-transformers faiss-cpu python-dotenv pypdf langchain-ollama

## Start the server (bash)
#ollama serve



# Load environment variables (not needed anymore)
# load_dotenv()

# No API key needed for Ollama
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Use free Hugging Face embeddings instead of OpenAIEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


###we could also test other possible embeddings and compare performance e.g "nomic-embed-text" from ollama

#from langchain_ollama import OllamaEmbeddings

#embeddings = OllamaEmbeddings(model="nomic-embed-text")
#vectorstore = FAISS.from_documents(texts, embeddings)

#mechanism to rebuild the vectorstore if needed in case something was changed with force_rebuild = True
force_rebuild = True
faiss_path = "FAISS_db_Orwell/RAG"

if force_rebuild and os.path.exists(faiss_path):
    import shutil
    shutil.rmtree(faiss_path)
    print("Deleted old FAISS index (force rebuild).")

if os.path.exists(faiss_path):
    vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
else:
    loader = PyPDFLoader("George_Orwell_1984.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    texts = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(faiss_path)

# Use local Llama 3.1 model via Ollama
llm = ChatOllama(model="llama3.1:8b", temperature=0.1)

## we could also test other chain_types such as "map_reduce" (sends each junk separately to the llm then summarizes results)
## or "refine" (sends each junk individually to the llm)

#we could also play around with the numbers of chunks retrieved to see how the result is influenced:
# e.g retriever = vectorstore.as_retriever(search_kwargs={"k": 6}

###Prompt template
from langchain.prompts import PromptTemplate

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a careful assistant. Use ONLY the context to answer.\n"
        'If the answer is not fully supported by the context, say: "I do not know based on the provided context."'
        "Always cite sources as [page X] after each claim.\n\n"
        "Answer the question based only on the following context:\n"
        "{context}\n"
        "---\n"
        "Answer the question based on the above context: {question}"
    ),
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents = True,
    chain_type_kwargs={"prompt": custom_prompt}
)

query = "When does Winston see Julia for the first time?"

#Print the prompt

docs = qa_chain.retriever.get_relevant_documents(query)

#add page number to each chunk

context_text = "\n\n".join(
    f"[page {d.metadata.get('page','?')}]\n{d.page_content}"
    for d in docs)


# Render your custom prompt manually (exactly what goes to the LLM)
rendered_prompt = custom_prompt.format(context=context_text, question=query)

print("\n--- FULL PROMPT SENT TO LLM ---\n")
print(rendered_prompt)
print("\n--- END OF PROMPT ---\n")

response = qa_chain.invoke(query)
print(response)

#sources of the response

#print("\nSOURCES:")
#for i, doc in enumerate(response["source_documents"], start=1):
    #src = doc.metadata.get("source")
    #page = doc.metadata.get("page")
    #print(f"[{i}] {src} (page {page})")
