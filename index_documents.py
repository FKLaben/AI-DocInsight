import gradio as gr
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

def process_pdfs(pdf_files, index_name):
    documents = []
    
    # Load each PDF file into a list of Document objects.
    for pdf_file in pdf_files:
        loader = PyMuPDFLoader(pdf_file.name)
        docs = loader.load()
        documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    
    # Split the documents into chunks
    texts = text_splitter.split_documents(documents)
    
    # Initialize embeddings with OllamaEmbeddings
    embeddings = OllamaEmbeddings(model="mistral-small:24b")
    
    # Create a FAISS index and save it as specified by user.
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(index_name)

    return f"FAISS index saved as {index_name}"

# Gradio interface setup
iface = gr.Interface(
    fn=process_pdfs,
    inputs=[
        gr.File(file_count="multiple", file_types=[".pdf"], label="Upload PDF Files"),
        gr.Textbox(label="Index Name")
    ],
    outputs="text",
    title="PDF to FAISS Index Converter",
    description="Upload multiple PDF files and specify an index name to save the generated FAISS index.",
)

# Launching the Gradio interface
iface.launch()
