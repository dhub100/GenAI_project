# Advanced RAG with LangChain, Hugging Face, and Ollama

This project demonstrates an **Advanced Retrieval-Augmented Generation (RAG)** pipeline built with **LangChain**, **Hugging Face embeddings**, and a locally running **Llama 3.1** model via **Ollama**.  

The system answers questions based on the content of *George Orwell’s 1984* and integrates several improvements beyond a basic RAG setup.

The script automatically performs document loading, semantic chunking, embedding, vector search, and context-based answering.

---

## Requirements

- Python 3.10 or higher  
- A free Hugging Face account with an access token  
- Ollama installed and available locally  
- The Ollama server must be running before starting the script  

---

## Installation

Install all required dependencies:

```bash
pip install langchain langchain-community langchain-ollama faiss-cpu sentence-transformers nltk
```

---

## Setup

1. Place the file **`George_Orwell_1984.pdf`** in the project directory.  
2. Ensure you have a **Hugging Face access token** (required for embedding models).  
3. Start the Ollama server locally:
   ```bash
   ollama serve
   ```
   or use the provided `.bat` file (e.g. `start_ollama.bat`).
4. Make sure the model is available locally:
   ```bash
   ollama pull llama3.1:8b
   ```

---

## Usage

Before running the script, make sure the **Ollama server** is running.  
You only need to start it **once per session**, and it must stay active while the script runs.

Start the Ollama server (in a separate terminal)
```bash
ollama serve
```
Then, run the Advanced RAG script in another terminal:
```bash
python advanced_rag.py
```

The script will:
- Load the PDF document  
- Apply **semantic chunking** (sentence-based splitting for coherent meaning units)  
- Create vector embeddings using a Hugging Face model (`sentence-transformers/all-MiniLM-L6-v2`)  
- Store and search chunks in a **FAISS** vector database  
- Use **Llama 3.1 (8B)** through Ollama to generate context-aware answers  

Example query:
```
What is the Junior Anti-Sex League Orwell is writing about?
```

---

## Output

When you run the script, you will see several progress messages in the console:

1. **Loading steps** – confirms that embeddings, the FAISS vector database, and the local Llama model were loaded correctly.  
2. **Query expansion** – the script generates 3–5 alternative versions of your question and prints them. These rephrased prompts are also available in a Python list (`expanded_list`) for further analysis or evaluation.  
3. **Retrieval & generation** – progress messages indicate when the system retrieves relevant chunks and when the final answer is being generated.  
4. **Final Answer** – a single, context-based, concise answer is printed at the end of the run.  
5. The final answer is also stored internally as a string variable (`final_answer`), which can be reused in code or saved to a file.

Example:
```
Preparing query expansion...
Expanded Prompts:
 - What is the purpose of the Junior Anti-Sex League in 1984
 - What role does the Junior Anti-Sex League play in Orwell’s dystopian society
...
Retrieving context and generating answer...
Done!

Final Answer:
The Junior Anti-Sex League is a propaganda organization that promotes the Party’s ideology...
```

---

## Key Improvements Compared to a Basic RAG

This project implements four practical enhancements to move from a **naive RAG** to an **advanced RAG** system:

1. **Semantic Chunking**  
   Text is split by meaning and sentence boundaries, not fixed lengths.  
   → Improves context quality and reduces semantic breaks.

2. **Query Expansion**  
   The model can reformulate or enrich user queries to find more relevant chunks.  
   → Increases retrieval accuracy.

3. **Context Compression**  
   Retrieved chunks are summarized or cleaned before being passed to the LLM.  
   → Reduces token usage and improves answer precision.

4. **Structured Prompting**  
   Clear prompt sections for system instruction, context, and question.  
   → Leads to more consistent and traceable model responses.

---

## License

This project is licensed under the MIT License.
