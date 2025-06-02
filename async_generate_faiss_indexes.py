import asyncio
from concurrent.futures import ThreadPoolExecutor
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
import json
import argparse
import os
import zipfile
import shutil

# def extract_text_from_pdf(file_path):
#     doc = fitz.open(file_path)
#     text = ""
#     for page in doc:
#         text += page.get_text()
#     return text


# Asynchronous function to process each document and create a FAISS index
async def process_pdfs(pdf_file):

    # renamee is the file getting renamed, pre is the part of file name before extension and ext is current extension
    pre, ext = os.path.splitext(pdf_file)

    index_name = f"{pre}.index"
    documents = []
    

    loader = PyMuPDFLoader(pdf_file)
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

    # Clean up the saved index directory
    shutil.rmtree(index_name)
    print(f"FAISS index saved as {index_name}.zip")

# Asynchronous function to process each document and create a FAISS index
# async def process_document(file_path):
#     if file_path.endswith('.pdf'):
#         documents = extract_text_from_pdf(file_path)
#     else:
#         with open(file_path, 'r', encoding='utf-8') as file:
#             documents = json.load(file)

#     loop = asyncio.get_event_loop()
#     with ThreadPoolExecutor() as executor:
#         future = loop.run_in_executor(executor, create_faiss_index, documents)
#         print(f"Processing {file_path}...")
#         index = await asyncio.wrap_future(future)
#         faiss.write_index(index, f"{file_path}.index")
#         print(f"{file_path} done.")

def create_faiss_index(documents):
    # Dummy implementation for FAISS index creation
    # Replace this with actual FAISS index creation logic
    return "FAISS Index"

# Main function to handle the list of document file paths
async def main(file_paths):
    tasks = []
    for file_path in file_paths:
        task = process_pdfs(file_path)
        tasks.append(task)

    await asyncio.gather(*tasks)

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate FAISS indexes from document files.')
    parser.add_argument('file_paths', nargs='+', help='List of file paths containing documents')
    args = parser.parse_args()

    asyncio.run(main(args.file_paths))
