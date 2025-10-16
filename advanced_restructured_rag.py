import os
from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from langchain.schema import Document
# from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize
from enum import Enum

class ChainType(Enum):
    STUFF = "stuff"
    REFINE = "refine"
    MAP_REDUCE = "map_reduce"

class AdvancedRAG:
    def __init__(self, document_path: str = "George_Orwell_1984.pdf",
                 faiss_path: str = "FAISS_db_Orwell/RAG",  # "FAISS_db_Orwell_nomic/RAG"
                 rebuild_faiss: bool = False,
                 chain_type: ChainType = ChainType.REFINE,
                 hugging_face_embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):

        # defining constants
        self.document_path = document_path
        self.faiss_path = faiss_path
        self.rebuild_faiss = rebuild_faiss
        self.chain_type = chain_type.value
        self.hugging_face_embeddings_model_name = hugging_face_embeddings_model_name

        # initial setup
        self.warnings_display()
        self.setup_nltk()
        self.embeddings = self.setup_embeddings()
        self.vectorstore = self.setup_FAISS_indexing(rebuild_faiss=self.rebuild_faiss)

        #### LLM & Retrieval chain ####
        self.llm = self.setup_local_llm()
        self.qa_chain = self.setup_retrieval_chain()


    def warnings_display(self):
        ## Install Ollama locally:
        # https://ollama.com/download

        ## Pull the model (bash)
        # ollama pull llama3.1:8b

        ## Start the server (bash)
        # ollama serve
        # ollama pull llama3.1: 8b
        # ollama pull nomic-embed-text

        print("Make sure the Ollama server is running (use 'ollama serve' in another terminal)\n.")
        print("The server must be active once per session before starting this script.\n")

    def setup_FAISS_indexing(self, rebuild_faiss: bool = False):
        # Control whether to rebuild the FAISS index ###
        # ´rebuild_faisse´ : set to True to force re-indexing (e.g., after changing chunking strategy)

        if os.path.exists(self.faiss_path) and not rebuild_faiss:
            print("Loading existing FAISS vector database...")
            vectorstore = FAISS.load_local(self.faiss_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            print("Creating new FAISS vector database from PDF...")
            loader = PyPDFLoader(self.document_path)
            documents = loader.load()
            # Combine all pages into a single string
            full_text = " ".join([doc.page_content for doc in documents])
            # Apply semantic chunking
            semantic_chunks = self.semantic_chunk_text(full_text, target_length=800, overlap_sentences=2)
            # Convert chunks to LangChain Document objects
            texts = [
                Document(
                    page_content=chunk,
                    metadata={
                        "source": self.document_path,
                        "chunk_id": i + 1
                    }
                )
                for i, chunk in enumerate(semantic_chunks)
            ]
            # Create FAISS vectorstore
            vectorstore = FAISS.from_documents(texts, self.embeddings)
            vectorstore.save_local(self.faiss_path)
            print("New FAISS database saved.")
        return vectorstore

    def setup_local_llm(self) -> None:
        """
        Set up local LLM (Llama 3.1 model via Ollama)
        """
        print("Loading local Llama 3.1 model via Ollama...")
        return ChatOllama(model="llama3.1:8b", temperature=0.1)

    def setup_embeddings(self):
        """
        Use free Hugging Face embeddings
        """
        print("Loading sentence-transformer embeddings...")
        # return OllamaEmbeddings(model="nomic-embed-text")
        return HuggingFaceEmbeddings(model_name=self.hugging_face_embeddings_model_name)

    def setup_retrieval_chain(self):
        """
        Setup retrieval chain based on class attributes llm,
        chain_type et vectorstore.
        """
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=self.chain_type,
            retriever=self.vectorstore.as_retriever()
            )

    def setup_nltk(self):
        """
        Set up NLTK tokenizer
        """
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')

    def semantic_chunk_text(self, text, target_length=800, overlap_sentences=2):
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

    def compress_context(self, retrieved_docs, query: str, max_sentences: int = 2):
        """
        Compresses retrieved chunks using the local LLM.
        Summarizes each document briefly based on the given query.
        """
        print("\nCompressing retrieved context...")
        compressed = []

        for i, doc in enumerate(retrieved_docs, start=1):
            text = doc.page_content.strip()
            if not text:
                continue

            prompt = (
                f"Summarize this text in {max_sentences} sentences, focusing only on information "
                f"relevant to answering the following question:\n\n"
                f"Question: {query}\n\n"
                f"Text:\n{text}\n\n"
                f"Summary:"
            )

            try:
                summary = self.llm.invoke(prompt).content.strip()
                compressed.append(summary)
                print(f"  Chunk {i} compressed.")
            except Exception as e:
                print(f"  Chunk {i} compression failed: {e}")

        print("Context compression complete.\n")
        return compressed

    def expand_query(self, original_query: str) -> tuple[str, list[str]]:
        """
        Expands the user's query using the local LLM. Returns:
        - A single combined string for the RAG input
        - A list of alternative rephrasings (for display/logging)
        """
        print("Expanding query...")

        prompt = (
            f"Rephrase the following question in 3–5 alternative ways using synonyms or related concepts. "
            f"Only return a simple comma-separated list of the alternative questions. "
            f"Do not add explanations or headings.\n\n"
            f"Question: {original_query}\n\n"
            f"List:"
        )

        try:
            expanded_text = self.get_direct_llm_answer(prompt)

            # Parse the list from the model output
            expanded_list = [q.strip() for q in expanded_text.split(",") if q.strip()]

            # Combine all rephrasings into one natural expanded query
            combined_expanded = (
                "Please answer the following related questions together as one: "
                + "; ".join(expanded_list)
            )

            return combined_expanded, expanded_list

        except Exception as e:
            print(f"Query expansion failed: {e}")
            return original_query, [original_query]  # fallback

    def get_direct_llm_answer(self, prompt):
        response = self.llm.invoke(prompt)
        answer = response.content.strip()
        if isinstance(answer, dict) and "result" in answer:
            final_answer = answer["result"]
        else:
            final_answer = answer
        return final_answer

    def answer_query(self, query: str):
        """
        Runs the full RAG pipeline:
        1. Expands the user's query.
        2. Retrieves context from FAISS.
        3. Compresses the retrieved context before passing it to the LLM.
        4. Returns the final, concise answer.
        """

        print("\nPreparing query expansion...")
        expanded_query, expanded_list = self.expand_query(query)

        print("\nExpanded Prompts:")
        for q in expanded_list:
            print(" -", q)

        print("\nRetrieving relevant context...")
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        retrieved_docs = retriever.invoke(expanded_query)
        print(f"Retrieved {len(retrieved_docs)} documents.")

        # --- Context Compression (your existing version) ---
        compressed_contexts = self.compress_context(retrieved_docs, query=expanded_query, max_sentences=2)

        # Combine all compressed summaries into one text block
        context_text = "\n".join(compressed_contexts)

        # --- Final Combined Prompt ---
        final_prompt = (
            "Answer concisely (max 3 sentences) and only based on the context below.\n\n"
            f"Context:\n{context_text}\n\nQuestion: {expanded_query}\nAnswer:"
        )

        print("\nGenerating final answer...")
        response = self.llm.invoke(final_prompt)
        final_answer = response.content.strip() if hasattr(response, "content") else str(response)

        print("\nFinal Answer:\n", final_answer)
        return final_answer


# Run only when executed directly (not when imported)
if __name__ == "__main__":
    query = "What is the Junior Anti-Sex League Orwell is writing about?"
    final_answer = AdvancedRAG().answer_query(query)
