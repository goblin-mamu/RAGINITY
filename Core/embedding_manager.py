"""
Manages embedding models for document processing and retrieval.
"""
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingManager:
    """
    Manages the embedding models used for document vectorization and retrieval.
    """

    def __init__(self, embedding_models):
        """
        Initialize the embedding manager with available models.

        Args:
            embedding_models (dict): Dictionary of available embedding models configurations
        """
        self.available_models = embedding_models
        self.current_model_name = None
        self.current_model = None

    def list_available_models(self):
        """
        Returns list of available embedding model names.
        """
        return list(self.available_models.keys())

    def get_model_info(self, model_name):
        """
        Get information about a specific embedding model.

        Args:
            model_name (str): Name of the model to retrieve info for

        Returns:
            dict: Model configuration dictionary
        """
        return self.available_models.get(model_name)

    def get_current_model_name(self):
        """
        Get the name of the currently loaded model.

        Returns:
            str: Name of the current model, or None if no model is loaded
        """
        return self.current_model_name

    def get_model_details(self, model_name):
        """
        Get detailed description of a model for display.

        Args:
            model_name (str): The name of the model to describe

        Returns:
            tuple: (description, dimensions, recommended_ram)
        """
        model_info = self.get_model_info(model_name)
        description = model_info.get("description", "No description available")
        dimensions = model_info.get("dimensions", "Unknown")
        recommended_ram = model_info.get("recommended_ram", "Unknown")

        return description, dimensions, recommended_ram

    def initialize_model(self, model_name):
        """
        Initialize and load a specific embedding model.

        Args:
            model_name (str): Name of the model to initialize

        Returns:
            HuggingFaceEmbeddings: The initialized embedding model
        """
        model_config = self.get_model_info(model_name)
        if not model_config:
            raise ValueError(f"Unknown embedding model: {model_name}")

        with st.spinner(f"🧠 Loading embedding model: {model_config['model_name']}..."):
            embeddings = HuggingFaceEmbeddings(
                model_name=model_config["model_name"],
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
            )

            self.current_model = embeddings
            self.current_model_name = model_config["model_name"]

            return embeddings

    def get_or_initialize_model(self, model_name):
        """
        Get the current model if it matches the requested one, or initialize a new one.

        Args:
            model_name (str): Name of the model to get or initialize

        Returns:
            HuggingFaceEmbeddings: The requested embedding model
        """
        model_config = self.get_model_info(model_name)
        if not model_config:
            raise ValueError(f"Unknown embedding model: {model_name}")

        if self.current_model_name != model_config["model_name"]:
            return self.initialize_model(model_name)

        return self.current_model
