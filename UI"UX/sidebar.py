"""
Sidebar component for Streamlit application.
"""
import psutil
import streamlit as st


class Sidebar:
    """
    Manages the sidebar UI for the RAG comparison application.
    """

    def __init__(self, llm_manager, embedding_manager):
        """
        Initialize the sidebar with managers.

        Args:
            llm_manager: LLM manager instance
            embedding_manager: Embedding manager instance
        """
        self.llm_manager = llm_manager
        self.embedding_manager = embedding_manager

    def render(self):
        """
        Render the sidebar UI.

        Returns:
            dict: Dictionary containing all selected parameters
        """
        with st.sidebar:
            # System specifications
            self._render_system_specs()

            # Embedding model selection
            embedding_model, embedding_config = self._render_embedding_selection()

            # LLM model selection
            llm_model, llm_config = self._render_llm_selection()

            # Advanced settings
            temperature, k_value, chunk_size, chunk_overlap = self._render_advanced_settings(llm_config)

            # Cache controls
            clear_cache = self._render_cache_controls()

            # Add local processing information
            self._render_local_info()

            # Return all selected parameters
            return {
                'embedding_model': embedding_model,
                'embedding_config': embedding_config,
                'llm_model': llm_model,
                'llm_config': llm_config,
                'temperature': temperature,
                'k_value': k_value,
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'clear_cache': clear_cache
            }

    def _render_system_specs(self):
        """
        Render system specifications.
        """
        st.markdown("### 🖥 System Specifications")
        col_sys1, col_sys2 = st.columns(2)

        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()

        with col_sys1:
            st.metric("CPU Cores", f"{cpu_count}")
        with col_sys2:
            st.metric("Total RAM", f"{memory.total / (1024 ** 3):.1f} GB")

        st.markdown("---")

    def _render_embedding_selection(self):
        """
        Render embedding model selection.

        Returns:
            tuple: (selected_model_name, model_config)
        """
        st.markdown("### 🧠 Embedding Model")

        # Get available models
        embedding_models = self.embedding_manager.list_available_models()

        # Set default model
        default_model = "intfloat/e5-base-v2 (Default)"
        if default_model in embedding_models:
            default_index = embedding_models.index(default_model)
        else:
            default_index = 0

        # Create selection box
        selected_model = st.selectbox(
            "Select Embedding Model",
            embedding_models,
            index=default_index
        )

        # Get model config
        embedding_config = self.embedding_manager.get_model_info(selected_model)

        # Display model details
        description, dimensions, _ = self.embedding_manager.get_model_details(selected_model)

        st.markdown(f"""
        **Model Description:**  
        {description}

        **Embedding Dimensions:** {dimensions}
        """)

        st.markdown("---")

        return selected_model, embedding_config

    def _render_llm_selection(self):
        """
        Render LLM model selection.

        Returns:
            tuple: (selected_model_name, model_config)
        """
        st.markdown("### 🔧 LLM Model Configuration")

        # Get available models
        llm_models = self.llm_manager.list_available_models()

        # Set default model
        default_model = "Neural-Chat 7B (Dialogue)"
        if default_model in llm_models:
            default_index = llm_models.index(default_model)
        else:
            default_index = 0

        # Create selection box
        selected_model = st.selectbox(
            "Select LLM Model",
            llm_models,
            index=default_index
        )

        # Get model config
        model_config = self.llm_manager.get_model_info(selected_model)

        # Display model details
        description, size, context_length, is_installed = self.llm_manager.get_model_details(selected_model)

        st.markdown(f"""
        **Model Description:**  
        {description}

        **Model Size:** {size}
        """)

        if not is_installed and model_config:
            model_id = model_config["id"]
            st.markdown(f"""
            <div class="alert alert-warning">
                <h4>⚠️ Model not installed</h4>
                <p>The selected model <code>{model_id}</code> doesn't appear to be installed in Ollama.</p>
                <p>It will be downloaded automatically when needed, or you can install it manually:</p>
                <code>ollama pull {model_id}</code>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 💡 Recommended Hardware")
        col_rec1, col_rec2 = st.columns(2)

        with col_rec1:
            st.metric("Recommended CPU", f"{model_config.get('recommended_cpu', 'N/A')}")
        with col_rec2:
            st.metric("Recommended RAM", f"{model_config.get('recommended_ram', 'N/A')} GB")

        st.markdown("---")

        return selected_model, model_config

    def _render_advanced_settings(self, model_config):
        """
        Render advanced settings.

        Args:
            model_config: LLM model configuration for default values

        Returns:
            tuple: (temperature, k_value, chunk_size, chunk_overlap)
        """
        st.markdown("### ⚙️ Advanced Settings")

        temperature = st.slider(
            "Temperature", 0.0, 1.0, 0.7, 0.1,
            help="Controls the randomness of responses. Lower values make output more deterministic and focused, while higher values introduce more creativity and variation."
        )

        k_value = st.slider(
            "Retrieved Chunks", 1, 5, 3,
            help="Number of document chunks to retrieve for each query. More chunks provide more context but can introduce noise."
        )

        # Chunking parameters
        st.markdown("### 📄 Document Processing")

        chunk_size = st.slider(
            "Chunk Size", 256, 2048, model_config.get("chunk_size", 512), 128,
            help="Size of text chunks in characters. Larger chunks preserve more context but may reduce retrieval precision."
        )

        chunk_overlap = st.slider(
            "Chunk Overlap", 0, 200, model_config.get("overlap", 50), 10,
            help="Number of overlapping characters between consecutive chunks. Higher overlap helps maintain context across chunk boundaries."
        )

        return temperature, k_value, chunk_size, chunk_overlap

    def _render_cache_controls(self):
        """
        Render cache control buttons.

        Returns:
            bool: True if cache should be cleared, False otherwise
        """
        return st.button("Clear Cache")

    def _render_local_info(self):
        """
        Render information about local processing.
        """
        st.markdown("---")
        with st.expander("🔒 How This System Works Locally"):
            st.markdown(
                """
                This RAG system operates completely locally on your machine:
                
                1. **Embedding Models**: HuggingFace models run locally
                2. **LLM Models**: Ollama runs inference locally
                3. **Vector Database**: FAISS runs in-memory
                4. **Document Processing**: PDF parsing happens locally
                
                No external API calls are made once models are downloaded,
                making it ideal for sensitive documents.
                """
            )
