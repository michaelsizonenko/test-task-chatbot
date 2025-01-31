import os
import time
from typing import Any
from uuid import uuid4

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone as PineconeClient
from pinecone import ServerlessSpec

from enums.prompt_templates import PromptTemplates
from services.config import config

os.environ["PINECONE_API_KEY"] = config.pinecone_api_key


class PdfRAG:
    def __init__(self, doc_folder_path: str) -> None:
        self.index_name = "new-index"
        self.doc_folder_path = doc_folder_path
        self.pc = PineconeClient()
        self.llm = ChatOpenAI(
            model_name="gpt-4o", temperature=0, openai_api_key=config.openai_key
        )
        self.embedding = OpenAIEmbeddings(openai_api_key=config.openai_key)
        self.chat_history: list[list[HumanMessage | Any]] = []

    def process_pdf(self) -> list[Document]:
        """Load the data and split it into chunks for processing into vectors.

        Returns:
            list[Document]: The split data.
        """
        loader = PyPDFDirectoryLoader(self.doc_folder_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(data)
        return docs

    def init_index(self) -> None:
        """Initialize the Pinecone index for keeping data."""
        existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]

        if self.index_name in existing_indexes:
            self.pc.delete_index(self.index_name)
        self.pc.create_index(
            name=self.index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not self.pc.describe_index(self.index_name).status["ready"]:
            time.sleep(1)
        index = self.pc.Index(self.index_name)
        documents = self.process_pdf()
        uuids = [str(uuid4()) for _ in range(len(documents))]
        vector_store = PineconeVectorStore(index=index, embedding=self.embedding)
        vector_store.add_documents(documents=documents, ids=uuids)

    def get_retriever(self) -> VectorStoreRetriever:
        """Get Pinecone vector retriever.

        Returns:
            VectorStoreRetriever: The retriever.
        """
        index = self.pc.Index(self.index_name)

        vector_store = PineconeVectorStore(index=index, embedding=self.embedding)

        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 1, "score_threshold": 0.5},
        )
        return retriever

    def get_answer(self, question: str) -> str:
        """Generate an answer to the question based on the documents.

        Args:
            question (str): The question to process.

        Returns:
            str: The answer.
        """
        custom_rag_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PromptTemplates.CONTEXT_RETRIEVAL_TEMPLATE),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PromptTemplates.CONTEXTUALIZE_SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.get_retriever(), contextualize_prompt
        )
        question_answer_chain = create_stuff_documents_chain(
            self.llm, custom_rag_prompt
        )

        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )

        response = rag_chain.invoke(
            {"input": question, "chat_history": self.chat_history}
        )

        self.chat_history.extend([HumanMessage(content=question), response["answer"]])
        return response["answer"]
