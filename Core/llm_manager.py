"""
Manages LLM models for RAG response generation.
"""
import requests
import streamlit as st
from langchain_ollama import OllamaLLM


class LLMManager:
    """
    Manages the Large Language Models used for generating responses.
    """

    def __init__(self, llm_models):
        """
        Initialize the LLM manager with available models.

        Args:
            llm_models (dict): Dictionary of available LLM configurations
        """
        self.available_models = llm_models
        self.current_model_name = None
        self.current_model = None
        self.installed_models = []
        self._check_installed_models()

    def _check_installed_models(self):
        """
        Check which models are already installed in Ollama.
        """
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.installed_models = [model["name"] for model in data.get("models", [])]
        except (requests.RequestException, ConnectionError):
            self.installed_models = []

    def is_ollama_running(self):
        """
        Check if Ollama service is running.

        Returns:
            bool: True if Ollama is running, False otherwise
        """
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except (requests.RequestException, ConnectionError):
            return False

    def is_model_installed(self, model_name):
        """
        Check if a specific model is installed.

        Args:
            model_name (str): Name of the model to check

        Returns:
            bool: True if the model is installed, False otherwise
        """
        model_config = self.get_model_info(model_name)
        if not model_config:
            return False
        return model_config["id"] in self.installed_models

    def list_available_models(self):
        """
        Returns list of available LLM model names.
        """
        return list(self.available_models.keys())

    def get_model_info(self, model_name):
        """
        Get information about a specific LLM model.

        Args:
            model_name (str): Name of the model to retrieve info for

        Returns:
            dict: Model configuration dictionary
        """
        return self.available_models.get(model_name)

    def get_model_details(self, model_name):
        """
        Get detailed description of a model for display.

        Args:
            model_name (str): The name of the model to describe

        Returns:
            tuple: (description, size, context_length, is_installed)
        """
        model_info = self.get_model_info(model_name)
        if not model_info:
            return None, None, None, False

        description = model_info.get("description", "No description available")
        size = model_info.get("size", "Unknown")
        context_length = model_info.get("context_length", "Unknown")
        is_installed = self.is_model_installed(model_name)

        return description, size, context_length, is_installed

    def initialize_model(self, model_name, temperature=0.7):
        """
        Initialize and load a specific LLM model.

        Args:
            model_name (str): Name of the model to initialize
            temperature (float): Temperature parameter for generation

        Returns:
            OllamaLLM: The initialized LLM model
        """
        model_config = self.get_model_info(model_name)
        if not model_config:
            raise ValueError(f"Unknown LLM model: {model_name}")

        with st.spinner(f"🤖 Loading LLM model: {model_config['id']}..."):
            llm = OllamaLLM(
                model=model_config["id"],
                temperature=temperature,
                context_window=model_config["context_length"],
                timeout=120
            )

            self.current_model = llm
            self.current_model_name = model_name

            return llm

    def get_or_initialize_model(self, model_name, temperature=0.7):
        """
        Get the current model if it matches the requested one, or initialize a new one.

        Args:
            model_name (str): Name of the model to get or initialize
            temperature (float): Temperature parameter for generation

        Returns:
            OllamaLLM: The requested LLM model
        """
        if self.current_model_name != model_name:
            return self.initialize_model(model_name, temperature)

        return self.current_model
