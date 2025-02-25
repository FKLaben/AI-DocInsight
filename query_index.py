import gradio as gr
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
import ollama
import re
import datetime

from tkinter import Tk, filedialog


from typing import IO, Any, cast

import chainlit as cl


def process(index_name):

    print('OllamaEmbeddings - {date:%Y-%m-%d_%H:%M:%S}'.format( date=datetime.datetime.now() ))
    embeddings = OllamaEmbeddings(model="mistral-small:24b")
    
    print('FAISS.from_documents - {date:%Y-%m-%d_%H:%M:%S}'.format( date=datetime.datetime.now() ))
    vectorstore = FAISS.load_local(index_name, embeddings, allow_dangerous_deserialization=True)
    
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
    return {result}
    

def get_folder_path(folder_path: str = "") -> str:
    """
    Opens a folder dialog to select a folder, allowing the user to navigate and choose a folder.
    If no folder is selected, returns the initially provided folder path or an empty string if not provided.
    This function is conditioned to skip the folder dialog on macOS or if specific environment variables are present,
    indicating a possible automated environment where a dialog cannot be displayed.

    Parameters:
    - folder_path (str): The initial folder path or an empty string by default. Used as the fallback if no folder is selected.

    Returns:
    - str: The path of the folder selected by the user, or the initial `folder_path` if no selection is made.

    Raises:
    - TypeError: If `folder_path` is not a string.
    - EnvironmentError: If there's an issue accessing environment variables.
    - RuntimeError: If there's an issue initializing the folder dialog.

    Note:
    - The function checks the `ENV_EXCLUSION` list against environment variables to determine if the folder dialog should be skipped, aiming to prevent its appearance during automated operations.
    - The dialog will also be skipped on macOS (`sys.platform != "darwin"`) as a specific behavior adjustment.
    - Credit: MalumaDev https://github.com/gradio-app/gradio/issues/2515#issuecomment-2393151550
    """
    # Validate parameter type
    if not isinstance(folder_path, str):
        raise TypeError("folder_path must be a string")

    try:
        root = Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        selected_folder = filedialog.askdirectory(initialdir=folder_path or ".")
        root.destroy()
        return selected_folder or folder_path
    except Exception as e:
        raise RuntimeError(f"Error initializing folder dialog: {e}") from e


def create_folder_ui(path="./"):
    with gr.Row():
        text_box = gr.Textbox(
            label="path",
            info="Path",
            lines=1,
            value=path,
        )
        button = gr.Button(value="\U0001f5c0", inputs=text_box, min_width=24)

        button.click(
            lambda: get_folder_path(text_box.value),
            outputs=[text_box],
        )

    return text_box, button

# Function to handle button click and update the folder path textbox
def select_folder_click():
    selected_path = get_folder_path()
    return selected_path

with gr.Blocks() as demo:
    with gr.Row():
        select_button = gr.Button("Select Folder", elem_id="select-folder-button")
        folder_textbox = gr.Textbox(label="Selected Folder Path")

        # Attach the button click event to update the textbox
        select_button.click(select_folder_click, inputs=[], outputs=[folder_textbox])
    
    with gr.Row():
        question_input = gr.Textbox(label="Enter your question here...", placeholder="What files are in this directory?")
        submit_btn = gr.Button("Submit")

    # Add question prompt and answer input
    with gr.Row():
        gr.Markdown("Response:")
        output_text = gr.Textbox(label="Response")

    submit_btn.click(fn=process_question, inputs=[folder_textbox, question_input], outputs=output_text)

# Launch the interface
demo.launch()
