# AI-DocInsight
AI-DocInsight is a document retrieval and question-answering system powered by Retrieval-Augmented Generation (RAG) and artificial intelligence/ This project enables users to efficiently index documents, retrieve relevant information, and generate insightful answers to their queries.

# Features 
- Document Indexing: Easily index a collection of documents to create a searchable knowledge base.
- Retrieval-Augmented Generation (RAG): Utilize RAG techniques to enhance the accuracy and relevance of retrieved information.
- User-Friendly Interface: A web-based interface allows users to select an index and submit questions effortlessly.
- AI-Powered Insights: Leverage AI to generate meaningful insights from indexed documents, providing valuable answers to user queries.

# Prerequisites 

     Python 3.8+

# Installation
```
conda env create -n langchain --file langchain.yml
pip install langchain langchain-community langchain-ollama langgraph transformers unstructured
conda install conda-forge::faiss-gpu
pip install gradio langchain_community pymupdf 
```

# Usage
## Indexing Documents:
Run the indexing script to create an index from your documents.
```
python .\index_documents.py
```

## Querying the Index:
Use the query script to select an index and submit questions.
```
python .\query_index.py
```
