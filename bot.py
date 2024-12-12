import os
import logging
import streamlit as st
from streamlit.logger import get_logger
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv
from utils import (
    create_vector_index,
)
from chains import (
    load_embedding_model,
    load_llm,
    configure_llm_only_chain,
    configure_qa_rag_chain,
    generate_ticket,
)
import asyncio

# Load environment variables from .env file
load_dotenv(".env")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Retrieve environment variables
url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM")

# Remapping for Langchain Neo4j integration
os.environ["NEO4J_URL"] = url

# Initialize Neo4j graph
try:
    neo4j_graph = Neo4jGraph(
        url=url, username=username, password=password, refresh_schema=False
    )
    create_vector_index(neo4j_graph)
except Exception as e:
    logger.error(f"Error initializing Neo4j graph: {e}")
    st.error("Error initializing Neo4j graph")
    raise

# Load embedding model
embeddings, dimension = load_embedding_model(
    embedding_model_name, config={"ollama_base_url": ollama_base_url}, logger=logger
)

# Load LLM
llm = load_llm(llm_name, logger=logger, config={"ollama_base_url": ollama_base_url})

# Configure LLM chains
llm_chain = configure_llm_only_chain(llm)
rag_chain = configure_qa_rag_chain(
    llm, embeddings, embeddings_store_url=url, username=username, password=password
)

class StreamHandler(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to Streamlit container."""

    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# Streamlit UI
styl = f"""
<style>
    /* not great support for :has yet (hello FireFox), but using it for now */
    .element-container:has([aria-label="Select RAG mode"]) {{
      position: fixed;
      bottom: 33px;
      background: white;
      z-index: 101;
    }}
    .stChatFloatingInputContainer {{
        bottom: 20px;
    }}

    /* Generate ticket text area */
    textarea[aria-label="Description"] {{
        height: 200px;
    }}

    .element-container:has([aria-label="What coding issue can I help you resolve today?"]) {{
        bottom: 45px;
    }} 
</style>
"""
st.markdown(styl, unsafe_allow_html=True)

def chat_input():
    """Handle user input and generate LLM response."""
    user_input = st.chat_input("What coding issue can I help you resolve today?")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            st.caption(f"RAG: {name}")
            stream_handler = StreamHandler(st.empty())
            try:
                result = asyncio.run(output_function(
                    {"question": user_input, "chat_history": []}, callbacks=[stream_handler]
                ))["answer"]
                output = result
                st.session_state[f"user_input"].append(user_input)
                st.session_state[f"generated"].append(output)
                st.session_state[f"rag_mode"].append(name)
            except Exception as e:
                logger.error(f"Error processing user input: {e}")
                st.error("Error processing user input")

def display_chat():
    """Display chat history."""
    # Session state
    if "generated" not in st.session_state:
        st.session_state[f"generated"] = []

    if "user_input" not in st.session_state:
        st.session_state[f"user_input"] = []

    if "rag_mode" not in st.session_state:
        st.session_state[f"rag_mode"] = []

    if st.session_state[f"generated"]:
        size = len(st.session_state[f"generated"])
        # Display only the last three exchanges
        for i in range(max(size - 3, 0), size):
            with st.chat_message("user"):
                st.write(st.session_state[f"user_input"][i])

            with st.chat_message("assistant"):
                st.caption(f"RAG: {st.session_state[f'rag_mode'][i]}")
                st.write(st.session_state[f"generated"][i])

        with st.expander("Not finding what you're looking for?"):
            st.write(
                "Automatically generate a draft for an internal ticket to our support team."
            )
            st.button(
                "Generate ticket",
                type="primary",
                key="show_ticket",
                on_click=open_sidebar,
            )
        with st.container():
            st.write("&nbsp;")

def mode_select() -> str:
    """Select RAG mode."""
    options = ["Disabled", "Enabled"]
    return st.radio("Select RAG mode", options, horizontal=True)

name = mode_select()
if name == "LLM only" or name == "Disabled":
    output_function = llm_chain
elif name == "Vector + Graph" or name == "Enabled":
    output_function = rag_chain

def open_sidebar():
    """Open sidebar for ticket generation."""
    st.session_state.open_sidebar = True

def close_sidebar():
    """Close sidebar for ticket generation."""
    st.session_state.open_sidebar = False

if not "open_sidebar" in st.session_state:
    st.session_state.open_sidebar = False
if st.session_state.open_sidebar:
    try:
        new_title, new_question = generate_ticket(
            neo4j_graph=neo4j_graph,
            llm_chain=llm_chain,
            input_question=st.session_state[f"user_input"][-1],
        )
        with st.sidebar:
            st.title("Ticket draft")
            st.write("Auto generated draft ticket")
            st.text_input("Title", new_title)
            st.text_area("Description", new_question)
            st.button(
                "Submit to support team",
                type="primary",
                key="submit_ticket",
                on_click=close_sidebar,
            )
    except Exception as e:
        logger.error(f"Error generating ticket: {e}")
        st.error("Error generating ticket")

display_chat()
chat_input()
