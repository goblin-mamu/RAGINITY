"""
Processes PDF documents for RAG system.
"""
import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """
    Handles document loading, parsing, and chunking.
    """

    def __init__(self):
        """
        Initialize the document processor.
        """
        self.processed_files = []

    def _is_already_processed(self, file_name, embedding_model, chunk_size, chunk_overlap):
        """
        Check if a file has already been processed with the same parameters.

        Args:
            file_name (str): Name of the file
            embedding_model (str): Name of the embedding model used
            chunk_size (int): Size of chunks used
            chunk_overlap (int): Overlap between chunks

        Returns:
            bool: True if the file was already processed, False otherwise
        """
        file_identifier = f"{file_name}_{embedding_model}_{chunk_size}_{chunk_overlap}"
        return file_identifier in self.processed_files

    def _mark_as_processed(self, file_name, embedding_model, chunk_size, chunk_overlap):
        """
        Mark a file as processed with specific parameters.

        Args:
            file_name (str): Name of the file
            embedding_model (str): Name of the embedding model used
            chunk_size (int): Size of chunks used
            chunk_overlap (int): Overlap between chunks
        """
        file_identifier = f"{file_name}_{embedding_model}_{chunk_size}_{chunk_overlap}"
        self.processed_files.append(file_identifier)

    def process_single_file(self, file_obj, chunk_size, chunk_overlap, embedding_model_name):
        """
        Process a single PDF file and split it into chunks.

        Args:
            file_obj: Streamlit file object
            chunk_size (int): Size of chunks in characters
            chunk_overlap (int): Overlap between chunks in characters
            embedding_model_name (str): Name of the embedding model used

        Returns:
            tuple: (documents, already_processed) - documents is list of document chunks,
                  already_processed is True if file was cached
        """
        if self._is_already_processed(file_obj.name, embedding_model_name, chunk_size, chunk_overlap):
            return [], True

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_obj.getvalue())
            tmp_path = tmp_file.name

        try:
            with st.spinner(f"Processing {file_obj.name}..."):
                # Load the PDF file
                loader = PDFPlumberLoader(tmp_path)
                documents = loader.load()

                # Add source information to metadata
                for doc in documents:
                    doc.metadata["source"] = file_obj.name

                # Split into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
                )

                chunks = text_splitter.split_documents(documents)

                # Mark file as processed
                self._mark_as_processed(file_obj.name, embedding_model_name, chunk_size, chunk_overlap)

                return chunks, False

        finally:
            os.unlink(tmp_path)

    def process_multiple_files(self, file_objs, chunk_size, chunk_overlap, embedding_model_name):
        """
        Process multiple PDF files.

        Args:
            file_objs (list): List of Streamlit file objects
            chunk_size (int): Size of chunks in characters
            chunk_overlap (int): Overlap between chunks in characters
            embedding_model_name (str): Name of the embedding model used

        Returns:
            tuple: (chunks, file_stats, processed_count)
        """
        all_chunks = []
        file_stats = {}
        processed_count = 0

        for file_obj in file_objs:
            chunks, was_cached = self.process_single_file(
                file_obj,
                chunk_size,
                chunk_overlap,
                embedding_model_name
            )

            if not was_cached:
                processed_count += 1

            if chunks:
                all_chunks.extend(chunks)
                file_stats[file_obj.name] = len(chunks)
            elif was_cached and file_obj.name in file_stats:
                # If cached, use existing stats
                pass

        return all_chunks, file_stats, processed_count

    def get_file_size(self, file_obj):
        """
        Get file size in MB.

        Args:
            file_obj: Streamlit file object

        Returns:
            float: Size in MB
        """
        return round(len(file_obj.getvalue()) / (1024 * 1024), 2)

    def clear_cache(self):
        """
        Clear the processing cache.
        """
        self.processed_files = []
