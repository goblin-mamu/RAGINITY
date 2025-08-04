"""
RAG Model Comparison System - Main Application

This application allows users to compare different RAG (Retrieval Augmented Generation)
models using Ollama for local inference and Hugging Face embeddings for document indexing.
"""
import time
import streamlit as st

from config.models_config import LLM_MODELS, EMBEDDING_MODELS, DEFAULT_RAG_PROMPT
from core import DocumentProcessor, EmbeddingManager, LLMManager, MetricsTracker, RAGManager, VectorStore
from ui import UIComponents, MetricsVisualization, Sidebar, StyleManager

# Set page configuration
st.set_page_config(
    page_title="RAG Model Comparison System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
StyleManager.apply_styles()


def initialize_session_state():
    """Initialize all session state variables."""
    if 'document_processor' not in st.session_state:
        st.session_state.document_processor = DocumentProcessor()

    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = VectorStore()

    if 'metrics_tracker' not in st.session_state:
        st.session_state.metrics_tracker = MetricsTracker()

    if 'embedding_manager' not in st.session_state:
        st.session_state.embedding_manager = EmbeddingManager(EMBEDDING_MODELS)

    if 'llm_manager' not in st.session_state:
        st.session_state.llm_manager = LLMManager(LLM_MODELS)

    if 'rag_manager' not in st.session_state:
        st.session_state.rag_manager = RAGManager(
            st.session_state.vector_store,
            st.session_state.metrics_tracker,
            prompt_template=DEFAULT_RAG_PROMPT
        )

    if 'ollama_checked' not in st.session_state:
        st.session_state.ollama_checked = False


def main():
    """Main application function."""
    # Initialize components
    initialize_session_state()

    # Get component instances from session state
    doc_processor = st.session_state.document_processor
    vector_store = st.session_state.vector_store
    metrics_tracker = st.session_state.metrics_tracker
    embedding_manager = st.session_state.embedding_manager
    llm_manager = st.session_state.llm_manager
    rag_manager = st.session_state.rag_manager

    # Display header
    UIComponents.header()

    # Check if Ollama is running
    if not st.session_state.ollama_checked:
        if not llm_manager.is_ollama_running():
            UIComponents.ollama_not_running_error()
            st.stop()
        else:
            st.session_state.ollama_checked = True

    # Render sidebar and get parameters
    sidebar = Sidebar(llm_manager, embedding_manager)
    params = sidebar.render()

    # Process clear cache request
    if params['clear_cache']:
        doc_processor.clear_cache()
        vector_store.reset()
        metrics_tracker.clear()
        st.success("Cache cleared successfully!")
        st.rerun()

    # File upload section
    uploaded_files = UIComponents.file_upload_area(max_files=3)

    if not uploaded_files:
        UIComponents.no_files_uploaded_info()
        st.stop()

    # Process button
    process_button = st.button("Process PDF Files")

    if process_button:
        # Check if embedding model changed
        current_embedding_model = params['embedding_model']
        current_embedding_model_name = params['embedding_config']['model_name']

        if vector_store.get_current_embedding_model() != current_embedding_model_name:
            vector_store.reset()

        # Process PDFs
        with st.spinner("🔄 Processing PDFs..."):
            try:
                initial_metrics = metrics_tracker.get_process_metrics(interval=0.5)
                start_time = time.time()

                # Get embedding model
                embeddings = embedding_manager.get_or_initialize_model(current_embedding_model)

                # Process documents
                chunks, file_chunks, processed_count = doc_processor.process_multiple_files(
                    uploaded_files,
                    params['chunk_size'],
                    params['chunk_overlap'],
                    current_embedding_model_name
                )

                # Create vector store
                vector_store.add_documents(chunks, embeddings, current_embedding_model_name)

                # Calculate metrics
                end_time = time.time()
                final_metrics = metrics_tracker.get_process_metrics(interval=0.5)
                processing_time = end_time - start_time
                memory_used = final_metrics["memory_mb"] - initial_metrics["memory_mb"]
                cpu_used = final_metrics["cpu_percent"]

                # Display results
                if processed_count > 0:
                    UIComponents.processing_success(
                        processed_count,
                        processing_time,
                        memory_used,
                        cpu_used,
                        current_embedding_model_name
                    )

                    # Show chunks per file
                    UIComponents.file_chunks_expander(file_chunks, uploaded_files)

                elif file_chunks:
                    st.info("All files were already processed. Using cached vectorstore.")
                    UIComponents.file_chunks_expander(file_chunks)

            except Exception as e:
                st.error(f"❌ Error processing PDFs: {str(e)}")
                st.info("Try a different embedding model or check the PDF files.")
                return

    # Query section - only show if vectorstore is initialized
    if vector_store.is_initialized():
        question = UIComponents.query_input()

        if question:
            # Check if embedding model has changed
            current_embedding_model_name = params['embedding_config']['model_name']
            stored_embedding_model = vector_store.get_current_embedding_model()

            if stored_embedding_model != current_embedding_model_name:
                UIComponents.embedding_model_changed_warning(current_embedding_model_name)
                st.stop()

            # Generate answer
            try:
                with st.spinner("Generating answer..."):
                    # Get LLM
                    llm = llm_manager.get_or_initialize_model(
                        params['llm_model'],
                        params['temperature']
                    )

                    # Setup progress indicators
                    placeholder, progress_bar, status_text, progress_phases = UIComponents.progress_animation()

                    # Simulamos actualización manual, más segura que usar hilos
                    for i in range(1, 20):
                        # Actualizar métricas
                        current = metrics_tracker.get_ollama_metrics(interval=0.1)
                        placeholder.markdown(
                            f"**Current CPU Usage:** {current['cpu_percent']:.1f}% | "
                            f"**Current Memory Usage:** {current['memory_mb']:.1f} MB"
                        )

                        # Actualizar barra de progreso
                        progress_value = min(0.95, i * 0.05)
                        progress_bar.progress(progress_value)

                        # Actualizar fase
                        phase_index = min(3, i // 5)
                        status_text.markdown(f"**Status:** {progress_phases[phase_index]}...")

                        # No bloqueamos el hilo principal por mucho tiempo
                        if i < 10:  # Primeras iteraciones más lentas para mostrar progreso
                            time.sleep(0.1)

                    # Use RAG manager for actual generation
                    response, metrics = rag_manager.generate_answer(
                        question,
                        llm,
                        current_embedding_model_name,
                        params['k_value']
                    )

                    # Update UI with the final progress
                    progress_bar.progress(1.0)
                    status_text.markdown("**Status:** Answer ready!")
                    time.sleep(0.5)

                    # Clear progress indicators
                    UIComponents.clear_progress_indicators(placeholder, progress_bar, status_text)

                    # Save metrics
                    metrics_tracker.save_metrics(
                        params['llm_model'],
                        current_embedding_model_name,
                        metrics['response_time'],
                        metrics['tokens'],
                        metrics['memory_used'],
                        metrics['cpu_used'],
                        question
                    )

                    # Display results
                    UIComponents.display_answer(response.get('result', ''))

                    # Display source document if available
                    if response.get('source_documents') and len(response['source_documents']) > 0:
                        UIComponents.display_source_document(response['source_documents'][0])

                    # Display metrics
                    MetricsVisualization.display_performance_metrics(metrics)

                    # Display comparisons if more than one model has been tried
                    if len(metrics_tracker.get_model_keys()) > 1:
                        comparison_data = MetricsVisualization.create_comparison_table(metrics_tracker)
                        MetricsVisualization.display_comparison_table(comparison_data)

                        # Display charts
                        fig = metrics_tracker.plot_comparison_charts()
                        if fig:
                            st.pyplot(fig)

            except Exception as e:
                st.error(f"❌ Error generating answer: {str(e)}")
                st.info("Try adjusting model parameters or using a different model.")


if __name__ == "__main__":
    main()
