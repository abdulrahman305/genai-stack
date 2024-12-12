import os
import logging
import streamlit as st
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Neo4jVector
from streamlit.logger import get_logger
from chains import (
    load_embedding_model,
    load_llm,
)
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(".env")

# Retrieve environment variables
url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM")

# Remapping for Langchain Neo4j integration
os.environ["NEO4J_URL"] = url

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load embedding model
embeddings, dimension = load_embedding_model(
    embedding_model_name, config={"ollama_base_url": ollama_base_url}, logger=logger
)


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


# Load LLM
llm = load_llm(llm_name, logger=logger, config={"ollama_base_url": ollama_base_url})


def main():
    st.header("📄Chat with your pdf file")

    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        try:
            pdf_reader = PdfReader(pdf)
        except Exception as e:
            logger.error(f"Error reading PDF file: {e}")
            st.error("Error reading PDF file")
            return

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Langchain text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )

        chunks = text_splitter.split_text(text=text)

        try:
            # Store the chunks part in db (vector)
            vectorstore = Neo4jVector.from_texts(
                chunks,
                url=url,
                username=username,
                password=password,
                embedding=embeddings,
                index_name="pdf_bot",
                node_label="PdfBotChunk",
                pre_delete_collection=True,  # Delete existing PDF data
            )
        except Exception as e:
            logger.error(f"Error storing chunks in Neo4j: {e}")
            st.error("Error storing chunks in Neo4j")
            return

        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
        )

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file")

        if query:
            stream_handler = StreamHandler(st.empty())
            try:
                qa.run(query, callbacks=[stream_handler])
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                st.error("Error processing query")


if __name__ == "__main__":
    main()
