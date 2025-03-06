import gradio as gr

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
import ollama
import re
import datetime


from typing import IO, Any, cast


import zipfile
import shutil


def process(index_name):

    print('OllamaEmbeddings - {date:%Y-%m-%d_%H:%M:%S}'.format( date=datetime.datetime.now() ))
    embeddings = OllamaEmbeddings(model="mistral-small:24b")
    

    # Extract the contents of the zip archive
    extract_path = f"{index_name}.unzipped"
    with zipfile.ZipFile(index_name, 'r') as zipf:
        zipf.extractall(extract_path)

    # Load the FAISS index from the extracted directory

    print('FAISS.from_documents - {date:%Y-%m-%d_%H:%M:%S}'.format( date=datetime.datetime.now() ))
    vectorstore = FAISS.load_local(extract_path, embeddings, allow_dangerous_deserialization=True)
    
    print('vectorstore.as_retriever() - {date:%Y-%m-%d_%H:%M:%S}'.format( date=datetime.datetime.now() ))
    retriever = vectorstore.as_retriever()
    print('end of process - {date:%Y-%m-%d_%H:%M:%S}'.format( date=datetime.datetime.now() ))
    return vectorstore, retriever



def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
    
def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    
    print('ollama.chat - {date:%Y-%m-%d_%H:%M:%S}'.format( date=datetime.datetime.now() ))
    response = ollama.chat(model="mistral-small:24b", messages=[{'role': 'user', 'content': formatted_prompt}])
    print('done! - {date:%Y-%m-%d_%H:%M:%S}'.format( date=datetime.datetime.now() ))
    response_content = response['message']['content']
    
    # Remove content between <think> and </think> tags to remove thinking output
    final_answer = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL).strip()

    return final_answer

def rag_chain(question, vectorstore, retriever):
    """
    Improved RAG chain to query and analyze documents using provided parameters.

    Args:
        question (str): The user's question to be answered
        vectorstore: Vector store containing document embeddings
        retriever: Retriever object for fetching relevant documents

    Returns:
        str: Final answer after querying and analyzing documents.
    """
    # Query the vector store to retrieve relevant documents
    docs = vectorstore.similarity_search(question, k=3)  # Search top 3 similar documents
    
    # Combine retrieved documents into context
    if not docs:
        return "No relevant documents found."
    
    # Split question into chunks (if needed)
    # question_chunks = text_splitter.split_text(question)
    
    # Pass combined context and question to LLM for analysis
    context = combine_docs(docs)
    answer = ollama_llm(context + "\nQuestion: " + question, context)
    
    return answer
    
def process_question(index_name, question):

    vectorstore, retriever = process(index_name)
    result = rag_chain(question, vectorstore, retriever)
    shutil.rmtree(f"{index_name}.unzipped")
    return {result}
    

def display_file_path(file):
    # Display the file path in a textbox
    return f"Selected file: {file}"

with gr.Blocks() as demo:
    with gr.Row():
        # File explorer component to select the ZIP file
        file_explorer = gr.FileExplorer(label="Select ZIP file")

        # Textbox to display the selected file path
        file_path_output = gr.Textbox(label="File Path", interactive=False)

    with gr.Row():
        # Textbox for the user to ask a question
        question_input = gr.Textbox(label="Ask a question about the ZIP file")

        # Button to trigger processing
        process_button = gr.Button("Process Question")

        # Output component for the processed question result
        output = gr.Textbox(label="Output", interactive=False)

    # Define the function to be called when the process button is clicked
    def on_process_button_click(file, question):
        if not file:
            return "Please select a file first.", ""
        result = process_question(file[0], question)
        return display_file_path(file[0]), result

    # Event listener for file selection change
    def on_file_selection_change(files):
        if files:
            return display_file_path(files[0])
        else:
            return ""

    # Event listeners
    file_explorer.change(on_file_selection_change, inputs=file_explorer, outputs=file_path_output)
    process_button.click(on_process_button_click, inputs=[file_explorer, question_input], outputs=[file_path_output, output])

# Launch the Gradio interface
demo.launch()
