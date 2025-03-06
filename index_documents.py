import gradio as gr
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
import os
import zipfile


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
    # Compress the saved index into a zip archive
    with zipfile.ZipFile(f"{index_name}.zip", 'w') as zipf:
        for root, dirs, files in os.walk(index_name):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, index_name)
                zipf.write(file_path, arcname)

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
