# Advanced RAG with LangChain, Ollama & Hugging Face Embeddings

This project demonstrates an **Advanced Retrieval-Augmented Generation (RAG)** pipeline built with **LangChain**, **HuggingFace embeddings**, and a locally running **Llama 3.1** model via **Ollama**. Optionally Ollama embeddings can also be used for a fully local setup.

The system answers questions based on the content of *George Orwell’s 1984* and integrates several improvements beyond a basic RAG setup.

The script automatically performs document loading, semantic chunking, embedding, vector search, and context-based answering.

------------------------------------------------------------------------

## Requirements

-   Python 3.10 or higher\\
-   Ollama installed and available locally
-   The Ollama server must be running before starting the script

------------------------------------------------------------------------

## Installation

Install all required dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Setup

1.  Place the file **`George_Orwell_1984.pdf`** in the RAG_Database directory.

2. Complete your .env file with your personal absolute path to the project. 

3.  Start the Ollama server locally:

    ``` bash
    ollama serve
    ```

4.  Make sure the models are available locally:

    ``` bash
    ollama pull llama3.1:8b
    ollama pull nomic-embed-text
    ```

------------------------------------------------------------------------

## Usage

Before running the scripts, make sure the **Ollama server** is running.\
You only need to start it **once per session**, and it must stay active while the script runs.

Start the Ollama server (in a separate terminal):

``` bash
ollama serve
```

Then, run the Advanced RAG script in another terminal:

``` bash
python -m RAGs.basic_rag
python -m RAGs.advanced_rag
```

The script will:

-   Load the PDF document
-   Apply **semantic chunking** (sentence-based splitting for coherent meaning units)
-   **Embeddings** are generated with Hugging Face by default, or with Ollama if selected
-   Store and search chunks in a **FAISS** vector database
-   Use **Llama 3.1 (8B)** through Ollama to generate context-aware answers

Example query:

```         
What is the Junior Anti-Sex League Orwell is writing about?
```

------------------------------------------------------------------------

## Output

When you run the script, you will see several progress messages in the console:

1.  **Loading steps** – confirms that embeddings, the FAISS vector database, and the local Llama model were loaded correctly.\
2.  **Query expansion** – the script generates 3–5 alternative versions of your question and prints them.\
3.  **Retrieval & compression** – relevant chunks are retrieved and optionally summarized before being passed to the LLM.\
4.  **Final answer generation** – the chosen structured prompt template is filled and executed.\
5.  **Final Answer** – a concise, context-based answer is printed at the end, including sources and the selected role.

Example:

```         
Preparing query expansion...
Expanded Prompts:
 - What is the purpose of the Junior Anti-Sex League in 1984
 - What role does the Junior Anti-Sex League play in Orwell’s dystopian society
...
Retrieving relevant context...
Compressing retrieved context...
Generating final answer...

Final Answer:
The Junior Anti-Sex League is a propaganda organization that promotes the Party’s ideology...

Sources:
 - book George Orwell 1984 p. 47 chunk 47-2

Role used: debate
```

------------------------------------------------------------------------

## Key Improvements Compared to a Basic RAG

This project implements several practical enhancements to move from a **naive RAG** to an **advanced RAG** system:

1.  **Semantic Chunking**\
    Text is split by meaning and sentence boundaries, not fixed lengths.\
    → Improves context quality and reduces semantic breaks.

2.  **Query Expansion**\
    The model reformulates or enriches user queries to find more relevant chunks.\
    → Increases retrieval accuracy.

3.  **Context Compression**\
    Retrieved chunks are summarized before being passed to the LLM.\
    → Reduces token usage and improves answer precision.

4.  **Structured Prompt Presets**\
    Predefined prompt templates for roles such as `default`, `academic`, `debate`, `psychology`, and `historical`.\
    → Allows consistent, role-specific analysis styles.

5.  **Configurable Chain Types**\
    Supports `stuff`, `refine`, and `map_reduce` chain modes for flexible retrieval QA behavior.

6.  **Switchable Embedding Backends**\
    Choose between **HuggingFace** or **Ollama** embeddings.\
    → Enables both CPU-only and fully local setups.

------------------------------------------------------------------------

## Evaluation of the RAG’s Performance

We evaluated how different RAG parameters affect the model’s performance. The detailed results can be found in the `evaluation` folder.

### Evaluation Method

To assess the quality of the RAG answers, we asked an LLM to rate each answer based on the official (gold) answer.\
The main question for the LLM was:\
Does the predicted answer contain the information from the gold answer — and not too many unrelated elements?

For comparison, we also tested a more traditional metric: the cosine similarity between the embeddings of the predicted and gold answers.\
However, we found that the LLM-based scoring gave results that matched human judgment more closely (see `evaluation/evaluation_testing.ipynb` for details).

### Question Categories

We defined several question types to test different aspects of the RAG system: - **Single event fact** – questions about a specific event or fact\
- **Ongoing events fact** – questions about current or continuous situations
- **Interpretation** – questions requiring reasoning or understanding beyond direct facts
- **Tricky** – questions related to the book, but whose answer is *not* in the book
- **Counterfactual robustness** – questions where the book contains incorrect information, to test if the model can notice it
- **Negative rejection** – questions that are not about the book at all

### Results and Observations

The results showed that changes in parameters such as **compression**, **number of retrieved chunks**, **embedding model**, or **retrieval chain type** had only a small impact on overall performance.

The **type of question**, however, had a much stronger influence. Some questions consistently received high scores, while others scored low across all parameter settings.\
This suggests that the **quality and clarity of the questions** play an important role in the evaluation and should be analyzed further in future work.

------------------------------------------------------------------------

## License

This project is licensed under the MIT License.