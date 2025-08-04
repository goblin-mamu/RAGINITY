"""
Reusable UI components for Streamlit application.
"""
import time
import threading
import streamlit as st


class UIComponents:
    """
    Common UI components for the RAG comparison application.
    """

    @staticmethod
    def header():
        """Render the application header."""
        st.title("📚 RAG Model Comparison System")

    @staticmethod
    def ollama_not_running_error():
        """Render error message when Ollama is not running."""
        st.markdown("""
        <div class="alert alert-danger">
            <h3>⚠️ Ollama is not running</h3>
            <p>This application requires Ollama to be installed and running.</p>
            <p>Please start Ollama service and refresh this page.</p>
            <ul>
                <li>To install Ollama, visit: <a href="https://ollama.com/download" target="_blank">https://ollama.com/download</a></li>
                <li>After installation, make sure the Ollama service is running before using this application.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def model_not_installed_warning(model_id):
        """
        Render warning when model is not installed.

        Args:
            model_id (str): ID of the model that is not installed
        """
        st.markdown(f"""
        <div class="alert alert-warning">
            <h4>⚠️ Model not installed</h4>
            <p>The selected model <code>{model_id}</code> doesn't appear to be installed in Ollama.</p>
            <p>It will be downloaded automatically when needed, or you can install it manually:</p>
            <code>ollama pull {model_id}</code>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def embedding_model_changed_warning(model_name):
        """
        Render warning when embedding model has changed.

        Args:
            model_name (str): Name of the new embedding model
        """
        st.warning(f"""
        ⚠️ Embedding model has changed!

        You're now using {model_name} for embeddings, but your documents 
        were processed with a different model. Please process your PDFs again with the new 
        embedding model for accurate results.
        """)

    @staticmethod
    def file_upload_area(max_files=3):
        """
        Render the file upload area.

        Args:
            max_files (int): Maximum number of files allowed

        Returns:
            list: List of uploaded file objects
        """
        st.markdown("### 📚 Document Knowledge Base")
        uploaded_files = st.file_uploader("Upload up to 3 PDF files", type="pdf", accept_multiple_files=True)

        if uploaded_files and len(uploaded_files) > max_files:
            st.warning(f"⚠️ For demo purposes, please limit uploads to {max_files} PDF files maximum.")
            uploaded_files = uploaded_files[:max_files]

        return uploaded_files

    @staticmethod
    def no_files_uploaded_info():
        """Render info message when no files are uploaded."""
        st.info("""
        👆 Upload PDF files to start. The system will process them locally and allow you to ask questions about their content.
        """)

    @staticmethod
    def processing_success(processed_count, processing_time, memory_used, cpu_used, embedding_model):
        """
        Render success message after processing files.

        Args:
            processed_count (int): Number of files processed
            processing_time (float): Processing time in seconds
            memory_used (float): Memory used in MB
            cpu_used (float): CPU usage percentage
            embedding_model (str): Name of the embedding model used
        """
        st.success(f"""
        ✅ PDF Processing Complete:
        - Files processed: {processed_count}
        - Processing time: {processing_time:.2f}s
        - Memory used: {memory_used:.1f}MB
        - CPU usage (instant): {cpu_used:.1f}%
        - Embedding model: {embedding_model}
        """)

    @staticmethod
    def file_chunks_expander(file_chunks, uploaded_files=None):
        """
        Render an expander showing document chunks.

        Args:
            file_chunks (dict): Dictionary mapping filenames to chunk counts
            uploaded_files (list, optional): List of file objects for size calculation
        """
        with st.expander("Show document chunks"):
            for filename, chunk_count in file_chunks.items():
                if uploaded_files:
                    # Find file object with matching name
                    matching_files = [f for f in uploaded_files if f.name == filename]
                    if matching_files:
                        file_size = round(len(matching_files[0].getvalue()) / (1024 * 1024), 2)
                        st.markdown(f"**{filename}** ({file_size} MB): {chunk_count} chunks")
                        continue

                st.markdown(f"**{filename}**: {chunk_count} chunks")

    @staticmethod
    def query_input():
        """
        Render the query input field.

        Returns:
            str: The query entered by the user
        """
        st.markdown("---")
        st.header("🤔 Ask Questions")
        return st.text_input("Enter your question about the documents:")

    @staticmethod
    def display_answer(answer_text):
        """
        Display the answer to a query.

        Args:
            answer_text (str): The answer text to display
        """
        st.markdown("### 📝 Answer:")
        st.markdown(answer_text)

    @staticmethod
    def display_source_document(source_doc):
        """
        Display source document information.

        Args:
            source_doc: Document object containing source information
        """
        if not source_doc:
            return

        max_preview_length = 200

        # Prepare preview text (shortened)
        doc_text = source_doc.page_content
        if len(doc_text) > max_preview_length:
            preview_text = doc_text[:max_preview_length] + "..."
        else:
            preview_text = doc_text

        with st.expander("View Source Document"):
            st.markdown("**Most Relevant Source:**")

            # Show source file name if available
            if hasattr(source_doc, 'metadata') and source_doc.metadata and 'source' in source_doc.metadata:
                source_file = source_doc.metadata['source']
                st.markdown(f"📄 **File:** {source_file}")

            st.markdown(f"```\n{preview_text}\n```")

            if hasattr(source_doc, 'metadata') and source_doc.metadata:
                st.markdown("**Metadata:**")
                for key, value in source_doc.metadata.items():
                    if key != 'source':  # Already displayed source above
                        st.markdown(f"- **{key}**: {value}")

    @staticmethod
    def display_performance_metrics(response_time, memory_used, cpu_used, tokens):
        """
        Display performance metrics.

        Args:
            response_time (float): Response time in seconds
            memory_used (float): Memory used in MB
            cpu_used (float): CPU usage percentage
            tokens (int): Number of tokens in the response
        """
        st.markdown("### 📊 Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Response Time", f"{response_time:.2f}s")
        with col2:
            st.metric("Avg Memory Used", f"{memory_used:.1f}MB")
        with col3:
            st.metric("Avg CPU Usage", f"{cpu_used:.1f}%")
        with col4:
            st.metric("Output Tokens", tokens)

    @staticmethod
    def display_model_comparison_table(metrics_data):
        """
        Display model comparison table.

        Args:
            metrics_data (list): List of dictionaries containing model metrics
        """
        if not metrics_data:
            return

        st.markdown("### 📈 Model Comparison")
        st.table(metrics_data)

    @staticmethod
    def display_comparison_charts(figure):
        """
        Display model comparison charts.

        Args:
            figure: Matplotlib figure to display
        """
        if figure:
            st.pyplot(figure)

    @staticmethod
    def progress_animation():
        """
        Display an animated progress bar during generation.
        Returns:
            tuple: (placeholder, progress_bar, status_text) - UI components to update
        """
        placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Define the phases for a natural progression feeling
        progress_phases = [
            "Retrieving context",
            "Processing information",
            "Generating response",
            "Finalizing answer"
        ]

        # No threading - we'll update manually from the main thread
        return placeholder, progress_bar, status_text, progress_phases

    @staticmethod
    def clear_progress_indicators(placeholder, progress_bar, status_text):
        """
        Clear progress indicators from the UI.

        Args:
            placeholder: Placeholder component
            progress_bar: Progress bar component
            status_text: Status text component
        """
        # Clear UI components
        placeholder.empty()
        progress_bar.empty()
        status_text.empty()
