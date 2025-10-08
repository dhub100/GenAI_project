# 📚 Basic RAG with LangChain and OpenAI

This project demonstrates **Retrieval-Augmented Generation (RAG)** using **LangChain** and **OpenAI**, based on a PDF document.

---

## ⚙️ Requirements

- **Python 3.10+**

### Install dependencies

```bash
pip install langchain langchain-community langchain-openai faiss-cpu python-dotenv
```

---

## 🚀 Usage

1. Place the file **`Schiller_Mary_Stuart.pdf`** in the project directory.  
2. Set the environment variable **`OPENAI_API_KEY`** (for example, in a `.env` file):
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
3. Run the script **`Basic_RAG.py`**:
   ```bash
   python Basic_RAG.py
   ```

---

## 🧠 How It Works

This script implements a **basic Retrieval-Augmented Generation (RAG)** workflow:

1. **Document Processing:**  
   The PDF file *Schiller_Mary_Stuart.pdf* is converted into text and split into smaller chunks.

2. **Vectorization & Storage:**  
   These text chunks are transformed into numerical vectors using OpenAI embeddings and stored in a **FAISS vector database** for efficient similarity search.

3. **Question Answering:**  
   When a user query is received, LangChain retrieves the most relevant text passages from the vector store and combines them with the prompt before sending them to the **OpenAI language model** (e.g., GPT-4).

4. **Response Generation:**  
   The model produces a context-aware answer that integrates the retrieved information with its own language understanding.

---

## 📄 License

This project is licensed under the **MIT License**.
