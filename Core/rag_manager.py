"""
Manages the RAG (Retrieval Augmented Generation) pipeline.
"""
import time
import threading
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


class RAGManager:
    """
    Manages the RAG (Retrieval Augmented Generation) pipeline.
    """

    def __init__(self, vector_store, metrics_tracker, prompt_template=None):
        """
        Initialize the RAG Manager.

        Args:
            vector_store: The vector store instance
            metrics_tracker: The metrics tracker instance
            prompt_template (str, optional): Custom prompt template to use
        """
        self.vector_store = vector_store
        self.metrics_tracker = metrics_tracker
        self.prompt_template = prompt_template

    def generate_answer(self, question, llm, embedding_model_name, k_value=3):
        """
        Generate an answer for a question using RAG.

        Args:
            question (str): The question to answer
            llm: The LLM to use for generation
            embedding_model_name (str): Name of the current embedding model (for tracking)
            k_value (int): Number of chunks to retrieve
            progress_callback (func, optional): Callback for progress updates (not used in thread-safe version)

        Returns:
            tuple: (result, metrics)
        """
        # Check if embedding model matches the one used for the vector store
        if self.vector_store.get_current_embedding_model() != embedding_model_name:
            raise ValueError(
                f"Current embedding model ({embedding_model_name}) doesn't match "
                f"the one used to create the vector store ({self.vector_store.get_current_embedding_model()})"
            )

        # Check if vectorstore is initialized
        if not self.vector_store.is_initialized():
            raise ValueError("Vector store is not initialized")

        start_time = time.time()

        # Get the retriever
        retriever = self.vector_store.get_retriever(k=k_value)

        # Setup the prompt template
        if not self.prompt_template:
            template = """
            Instructions: Use the provided context to answer the question clearly and directly.
            If you cannot find the answer in the context, simply state that you cannot answer based on the available information.
            Provide your response in a clear, professional manner. Include citations or references to the document when appropriate.

            Context: {context}

            Question: {question}

            Answer:"""
        else:
            template = self.prompt_template

        # Create the RAG chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template=template,
                    input_variables=["context", "question"]
                ),
            }
        )

        # Set up monitoring
        stop_event, samples = self.metrics_tracker.start_monitoring()

        # Run the chain in a separate thread to allow for monitoring
        result_container = {}

        def run_chain():
            result_container["result"] = qa_chain.invoke(question)

        chain_thread = threading.Thread(target=run_chain)
        chain_thread.start()
        chain_thread.join()

        # Stop monitoring
        stop_event.set()

        # Calculate metrics
        end_time = time.time()
        response_time = end_time - start_time

        # Get metrics from session state if available (more reliable in Streamlit)
        if hasattr(st.session_state, 'metrics_samples') and st.session_state.metrics_samples:
            cpu_samples = st.session_state.metrics_samples.get('cpu', [])
            memory_samples = st.session_state.metrics_samples.get('memory', [])
        else:
            cpu_samples = samples.get('cpu', [])
            memory_samples = samples.get('memory', [])

        avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
        avg_memory = sum(memory_samples) / len(memory_samples) if memory_samples else 0.0

        # Get the result
        response = result_container.get("result", {})

        # Store metrics
        metrics = {
            "response_time": response_time,
            "cpu_used": avg_cpu,
            "memory_used": avg_memory,
            "tokens": len(response.get("result", "").split()) if response else 0
        }

        return response, metrics
