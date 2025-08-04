"""
Manages vector storage for document embeddings.
"""
import streamlit as st
from langchain_community.vectorstores import FAISS


class VectorStore:
    """
    Manages the vector database for document embeddings and retrieval.
    """

    def __init__(self):
        """
        Initialize the vector store.
        """
        self.vectorstore = None
        self.current_embedding_model = None

    def get_current_embedding_model(self):
        """
        Get the name of the current embedding model.

        Returns:
            str: Name of current embedding model, or None if not set
        """
        return self.current_embedding_model

    def is_initialized(self):
        """
        Check if the vector store is initialized.

        Returns:
            bool: True if initialized, False otherwise
        """
        return self.vectorstore is not None

    def create_from_documents(self, documents, embeddings, embedding_model_name):
        """
        Create a new vector store from documents.

        Args:
            documents (list): List of document chunks
            embeddings: Embedding model to use
            embedding_model_name (str): Name of the embedding model for tracking

        Returns:
            VectorStore: Self for method chaining
        """
        with st.spinner("📄 Creating vector embeddings for document chunks..."):
            self.vectorstore = FAISS.from_documents(
                documents,
                embeddings,
                distance_strategy="METRIC_INNER_PRODUCT"
            )
            self.current_embedding_model = embedding_model_name

        return self

    def add_documents(self, documents, embeddings, embedding_model_name):
        """
        Add new documents to an existing vector store.

        Args:
            documents (list): List of document chunks
            embeddings: Embedding model to use
            embedding_model_name (str): Name of the embedding model for tracking

        Returns:
            VectorStore: Self for method chaining
        """
        # Check if embedding model changed
        if self.current_embedding_model != embedding_model_name:
            return self.create_from_documents(documents, embeddings, embedding_model_name)

        # If vectorstore doesn't exist, create it
        if not self.vectorstore:
            return self.create_from_documents(documents, embeddings, embedding_model_name)

        # Add documents to existing vectorstore
        if documents:
            with st.spinner("📄 Adding new documents to vector store..."):
                self.vectorstore.add_documents(documents)

        return self

    def get_retriever(self, k=3):
        """
        Get a retriever for the vector store.

        Args:
            k (int): Number of documents to retrieve

        Returns:
            Retriever: Retriever object
        """
        if not self.vectorstore:
            raise ValueError("Vector store is not initialized")

        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

    def reset(self):
        """
        Reset the vector store.
        """
        self.vectorstore = None
        self.current_embedding_model = None
