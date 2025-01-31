import os
import tempfile
import time
from typing import Generator

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from services.rag import PdfRAG


class RAGChatbot:
    """Class for a Streamlit-based chatbot to accept PDF files and answer
    questions on them.
    """

    def __init__(self) -> None:
        self.doc_path: str | None = None
        self.messages = st.session_state.get("messages", [])
        self.pdfrag = st.session_state.get("pdf_rag", None)
        self.temp_folder = st.session_state.get(
            "temp_folder", tempfile.TemporaryDirectory()
        )

    @staticmethod
    def stream_data(text: str) -> Generator[str, None, None]:
        """Present text in a "streaming" manner.

        Args:
            text (str): The text to display.

        Yields:
            Generator[str, None, None]: The streamed text.
        """
        for word in text.split(" "):
            yield word + " "
            time.sleep(0.02)

    def process_messages(self, pdfrag: PdfRAG) -> None:
        """Process the questions and extract the answer from the uploaded document.

        Args:
            pdfrag (PdfRAG): Instance of a document processor.
        """
        for message in self.messages:
            st.chat_message("human").write(message[0])
            st.chat_message("ai").write(message[1])

        if query := st.chat_input():
            st.chat_message("human").write(query)
            response = pdfrag.get_answer(query)
            st.chat_message("ai").write_stream(self.stream_data(response))
            self.messages.append([query, response])
            st.session_state["messages"] = self.messages

    def back_button_callback(self) -> None:
        """Handle "Back" button callback."""
        with st.spinner("Wait for it..."):
            self.messages = []
            st.session_state["messages"] = self.messages
            st.session_state.temp = tempfile.TemporaryDirectory()
            self.temp_folder = tempfile.TemporaryDirectory()
            st.session_state["temp_folder"] = self.temp_folder
            st.session_state.clicked = False

    def ingest_file(self, uploaded_files: list[UploadedFile]) -> None:
        """Handle new files being added.
        Create a new PdfRAG instance and process new files into vectors.

        Args:
            uploaded_files (list[UploadedFile]): The new files to process.
        """
        with st.spinner("Wait for it..."):
            for uploaded_file in uploaded_files:
                path = os.path.join(self.temp_folder.name, uploaded_file.name)
                with open(path, "wb") as f:
                    f.write(uploaded_file.getvalue())
            self.pdfrag = PdfRAG(self.temp_folder.name)
            st.session_state["pdf_rag"] = self.pdfrag
            self.pdfrag.init_index()
        st.session_state.clicked = True

    def run(self) -> None:
        """Run the chatbot."""
        st.set_page_config(page_title="RAG ChatBot")
        st.title("RAG Chatbot")
        st.markdown(
            "This chatbot aims to answer user questions based on PDF files "
            "the user uploads. Chatbot uses the RAG technique to fetch relevant "
            "context from documents to answer questions. "
        )

        if "clicked" not in st.session_state:
            st.session_state.clicked = False

        if not st.session_state.clicked:
            st.markdown("Choose a document to query:")
            uploaded_files = st.file_uploader(
                "Choose a PDF file", accept_multiple_files=True, type="pdf"
            )
            st.button(
                "Upload",
                type="primary",
                on_click=self.ingest_file,
                args=[uploaded_files],
            )

        if st.session_state.clicked:
            st.button("Back", type="primary", on_click=self.back_button_callback)
            self.process_messages(pdfrag=self.pdfrag)


if __name__ == "__main__":
    chatbot = RAGChatbot()
    chatbot.run()
