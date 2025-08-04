import os
import time
import psutil
import requests
import tempfile
import datetime
import threading

import streamlit as st
import matplotlib.pyplot as plt

from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Page configuration
st.set_page_config(
    page_title="RAG Model Comparison System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced custom CSS for better visibility
st.markdown("""
    <style>
    /* Main content area */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #262730;
        color: #FFFFFF;
    }

    /* Headers and metrics */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"],
    .stMetricLabel {
        color: #FFFFFF !important;
    }

    /* Table styling */
    .dataframe {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .dataframe th {
        background-color: #2E2E2E;
        color: #FFFFFF;
    }

    /* Alert box styling */
    .alert {
        padding: 12px;
        border-radius: 4px;
        margin-bottom: 16px;
    }
    .alert-danger {
        background-color: rgba(248, 113, 113, 0.2);
        border-left: 4px solid #ef4444;
        color: #fca5a5;
    }
    .alert-warning {
        background-color: rgba(251, 191, 36, 0.2);
        border-left: 4px solid #f59e0b;
        color: #fcd34d;
    }
    .alert-info {
        background-color: rgba(59, 130, 246, 0.2);
        border-left: 4px solid #3b82f6;
        color: #93c5fd;
    }
    </style>
""", unsafe_allow_html=True)

# Model configuration with recommended hardware (adjust as needed)
MODELS = {
    "Neural-Chat 7B (Dialogue)": {
        "id": "neural-chat:7b",
        "size": "4.1 GB",
        "context_length": 8192,
        "chunk_size": 512,
        "overlap": 50,
        "description": "Optimized for conversational responses",
        "recommended_cpu": 8,
        "recommended_ram": 16
    },
    "DeepSeek-R1 1.5B (Fast)": {
        "id": "deepseek-r1:1.5b",
        "size": "1.1 GB",
        "context_length": 8192,
        "chunk_size": 512,
        "overlap": 50,
        "description": "Fastest model, good for quick iterations",
        "recommended_cpu": 4,
        "recommended_ram": 8
    },
    "DeepSeek-R1 7B (Balanced)": {
        "id": "deepseek-r1:7b",
        "size": "4.7 GB",
        "context_length": 8192,
        "chunk_size": 1024,
        "overlap": 120,
        "description": "Strong reasoning capabilities",
        "recommended_cpu": 8,
        "recommended_ram": 16
    },
    "Mistral 7B (Quality)": {
        "id": "mistral:7b",
        "size": "4.1 GB",
        "context_length": 8192,
        "chunk_size": 512,
        "overlap": 50,
        "description": "High quality responses",
        "recommended_cpu": 8,
        "recommended_ram": 16
    },
    "Llama3 8B (General)": {
        "id": "llama3:8b",
        "size": "4.7 GB",
        "context_length": 4096,
        "chunk_size": 1024,
        "overlap": 120,
        "description": "Good all-round performance",
        "recommended_cpu": 8,
        "recommended_ram": 16
    },
    # Additional models
    "Phi3.5 (Small & Fast)": {
        "id": "phi3.5:3.8b",
        "size": "2.2 GB",
        "context_length": 4096,
        "chunk_size": 512,
        "overlap": 50,
        "description": "Lightweight model with great performance for its size",
        "recommended_cpu": 4,
        "recommended_ram": 8
    },
    "Gemma2 9B (Quality)": {
        "id": "gemma2:9b",
        "size": "5.2 GB",
        "context_length": 8192,
        "chunk_size": 1024,
        "overlap": 120,
        "description": "Google's high-quality model with good reasoning",
        "recommended_cpu": 8,
        "recommended_ram": 16
    },
    "DeepSeek-R1 MoE (Research)": {
        "id": "deepseek-r1:8b",
        "size": "4.8 GB",
        "context_length": 8192,
        "chunk_size": 1024,
        "overlap": 100,
        "description": "Enhanced model with strong reasoning capabilities",
        "recommended_cpu": 8,
        "recommended_ram": 16
    },
    "Qwen2.5 7B (Multilingual)": {
        "id": "qwen2.5:7b",
        "size": "4.1 GB",
        "context_length": 32768,
        "chunk_size": 1024,
        "overlap": 120,
        "description": "Excellent multilingual capabilities with long context",
        "recommended_cpu": 8,
        "recommended_ram": 16
    },
    "OpenCoder 8B (Code)": {
        "id": "opencoder:8b",
        "size": "4.7 GB",
        "context_length": 16384,
        "chunk_size": 1024,
        "overlap": 120,
        "description": "Specialized for code and technical documentation",
        "recommended_cpu": 8,
        "recommended_ram": 16
    }
}

# Hugging Face embedding models
EMBEDDING_MODELS = {
    "intfloat/e5-base-v2 (Default)": {
        "model_name": "intfloat/e5-base-v2",
        "description": "Base model with good performance for general text",
        "dimensions": 768,
        "recommended_ram": 2
    },
    "BAAI/bge-m3": {
        "model_name": "BAAI/bge-m3",
        "description": "Top ranked embedding model with multilingual support",
        "dimensions": 1024,
        "recommended_ram": 2
    },
    "mixedbread-ai/mxbai-embed-large-v1": {
        "model_name": "mixedbread-ai/mxbai-embed-large-v1",
        "description": "High performance model for precise retrieval",
        "dimensions": 768,
        "recommended_ram": 2
    },
    "Snowflake/arctic-embed-s": {
        "model_name": "Snowflake/arctic-embed-s",
        "description": "Lightweight and fast, good for quick iterations",
        "dimensions": 384,
        "recommended_ram": 1
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "Very small but effective for simple tasks",
        "dimensions": 384,
        "recommended_ram": 1
    }
}

# Initialize session state
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None

if 'llm' not in st.session_state:
    st.session_state.llm = None

if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

if 'uploaded_pdfs' not in st.session_state:
    st.session_state.uploaded_pdfs = {}

if 'processed_pdfs' not in st.session_state:
    st.session_state.processed_pdfs = []

if 'installed_models' not in st.session_state:
    st.session_state.installed_models = []

if 'ollama_checked' not in st.session_state:
    st.session_state.ollama_checked = False


def is_ollama_running():
    """Check if Ollama service is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except (requests.RequestException, ConnectionError):
        return False


def get_installed_ollama_models():
    """Get list of installed Ollama models"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            # Extract model IDs from the response
            models = [model["name"] for model in data.get("models", [])]
            return models
        return []
    except (requests.RequestException, ConnectionError):
        return []


def is_model_installed(model_id):
    """Check if specific model is installed in Ollama"""
    return model_id in st.session_state.installed_models


def initialize_embeddings(model_config):
    """Initialize the embedding model with optimal settings"""
    with st.spinner(f"🧠 Loading embedding model: {model_config['model_name']}..."):
        return HuggingFaceEmbeddings(
            model_name=model_config["model_name"],
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
        )


def initialize_llm(model_config, temperature=0.7):
    """Initialize the LLM with optimized parameters"""
    with st.spinner(f"🤖 Loading LLM model: {model_config['id']}..."):
        return OllamaLLM(
            model=model_config["id"],
            temperature=temperature,
            context_window=model_config["context_length"],
            timeout=120
        )


def process_pdfs(uploaded_files, chunk_size, chunk_overlap, embedding_model):
    """Process multiple PDFs and create a combined vectorstore"""
    all_splits = []
    processed_count = 0
    file_counts = {}

    for uploaded_file in uploaded_files:
        # Check if file already processed with current embedding model
        file_identifier = f"{uploaded_file.name}_{embedding_model['model_name']}_{chunk_size}_{chunk_overlap}"

        if file_identifier in st.session_state.processed_pdfs:
            st.info(f"Using cached data for {uploaded_file.name}")
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                loader = PDFPlumberLoader(tmp_path)
                documents = loader.load()

                # Add source information to metadata
                for doc in documents:
                    doc.metadata["source"] = uploaded_file.name

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
                )

                doc_splits = text_splitter.split_documents(documents)
                all_splits.extend(doc_splits)
                file_counts[uploaded_file.name] = len(doc_splits)
                processed_count += 1

                # Mark as processed
                st.session_state.processed_pdfs.append(file_identifier)

        finally:
            os.unlink(tmp_path)

    if not all_splits:
        return st.session_state.vectorstore, file_counts, 0

    # Use the selected embedding model
    embeddings = initialize_embeddings(embedding_model)

    # Show progress indicator during vectorization
    with st.spinner("📄 Creating vector embeddings for document chunks..."):
        if st.session_state.vectorstore is None:
            # Create new vectorstore
            vectorstore = FAISS.from_documents(
                all_splits,
                embeddings,
                distance_strategy="METRIC_INNER_PRODUCT"
            )
        else:
            # Add to existing vectorstore
            st.session_state.vectorstore.add_documents(all_splits)
            vectorstore = st.session_state.vectorstore

    # Update session state
    st.session_state.vectorstore = vectorstore

    return vectorstore, file_counts, processed_count


def process_pdf(uploaded_file, chunk_size, chunk_overlap, embedding_model):
    """Process the PDF using optimized chunking parameters"""
    # Check if file already processed with current embedding model
    file_identifier = f"{uploaded_file.name}_{embedding_model['model_name']}_{chunk_size}_{chunk_overlap}"

    if file_identifier in st.session_state.processed_files:
        st.info(f"Using cached vectorstore for {uploaded_file.name}")
        return st.session_state.vectorstore, st.session_state.num_chunks

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    try:
        loader = PDFPlumberLoader(tmp_path)
        documents = loader.load()

        # Add source information to metadata
        for doc in documents:
            doc.metadata["source"] = uploaded_file.name

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        splits = text_splitter.split_documents(documents)

        # Use the selected embedding model
        embeddings = initialize_embeddings(embedding_model)

        # Show progress indicator during vectorization
        with st.spinner("📄 Creating vector embeddings for document chunks..."):
            vectorstore = FAISS.from_documents(
                splits,
                embeddings,
                distance_strategy="METRIC_INNER_PRODUCT"
            )

        # Cache results
        st.session_state.vectorstore = vectorstore
        st.session_state.num_chunks = len(splits)
        st.session_state.processed_files.append(file_identifier)

        return vectorstore, len(splits)
    finally:
        os.unlink(tmp_path)


def get_process_metrics(interval=0.1):
    """
    Returns the metrics of the current process:
      - memory_mb: Memory used (RSS) in MB.
      - cpu_percent: CPU usage (percentage) measured over the given interval.
    """
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent = process.cpu_percent(interval=interval)
    return {"memory_mb": memory_mb, "cpu_percent": cpu_percent}


def get_ollama_metrics(interval=0.1):
    """
    Returns aggregated metrics for processes related to Ollama:
      - memory_mb: Total memory used (RSS) in MB.
      - cpu_percent: Sum of CPU usage (percentage) from each process, measured over the given interval.
    """
    matching = []
    for proc in psutil.process_iter(['name']):
        try:
            if "ollama" in proc.name().lower():
                matching.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
            print(f"Error accessing process: {e}")
            pass

    # Reset CPU counters for each matching process
    for p in matching:
        try:
            p.cpu_percent()
        except Exception as e:
            print(f"Error resetting CPU counter for PID {p.pid}: {e}")
            pass
    time.sleep(interval)
    cpu_sum = 0.0
    mem_sum = 0.0
    for p in matching:
        try:
            cpu_val = p.cpu_percent()
            cpu_sum += cpu_val
        except Exception as e:
            print(f"Error reading CPU for PID {p.pid}: {e}")
            pass
        try:
            mem = p.memory_info().rss
            mem_sum += mem / (1024 * 1024)
        except Exception as e:
            print(f"Error reading memory for PID {p.pid}: {e}")
            pass
    return {"cpu_percent": cpu_sum, "memory_mb": mem_sum}


def save_metrics(model_name, embedding_model_name, response_time, tokens, memory_used, cpu_used, query):
    """Save performance metrics for comparison"""
    # Create a unique key that combines LLM model and embedding model
    model_key = f"{model_name} + {embedding_model_name.split('/')[-1]}"

    if model_key not in st.session_state.model_metrics:
        st.session_state.model_metrics[model_key] = []

    st.session_state.model_metrics[model_key].append({
        'timestamp': datetime.datetime.now().strftime("%H:%M:%S"),
        'llm_model': model_name,
        'embedding_model': embedding_model_name,
        'response_time': response_time,
        'tokens': tokens,
        'memory_used': memory_used,
        'cpu_used': cpu_used,
        'query': query
    })


def plot_comparison_charts():
    """Generate comparison charts for models"""
    if len(st.session_state.model_metrics) < 1:
        return

    # Prepare data for plotting
    model_keys = list(st.session_state.model_metrics.keys())
    resp_times = []
    mem_usage = []
    cpu_usage = []

    for model_key in model_keys:
        metrics = st.session_state.model_metrics[model_key]
        resp_times.append(sum(m['response_time'] for m in metrics) / len(metrics))
        mem_usage.append(sum(m['memory_used'] for m in metrics) / len(metrics))
        cpu_usage.append(sum(m['cpu_used'] for m in metrics) / len(metrics))

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0E1117')

    # Shorten model names for better readability
    short_model_names = [name[:20] + '...' if len(name) > 20 else name for name in model_keys]

    # Response time plot
    ax1.bar(short_model_names, resp_times, color='skyblue')
    ax1.set_title('Average Response Time (s)', color='white')
    ax1.set_ylabel('Seconds', color='white')
    ax1.tick_params(axis='x', rotation=45, colors='white')
    ax1.tick_params(axis='y', colors='white')
    ax1.set_facecolor('#1E1E1E')

    # Set y-axis minimum to 0
    ax1.set_ylim(bottom=0)

    # Memory usage plot
    ax2.bar(short_model_names, mem_usage, color='lightgreen')
    ax2.set_title('Average Memory Usage (MB)', color='white')
    ax2.set_ylabel('MB', color='white')
    ax2.tick_params(axis='x', rotation=45, colors='white')
    ax2.tick_params(axis='y', colors='white')
    ax2.set_facecolor('#1E1E1E')

    # Set y-axis minimum to 0
    ax2.set_ylim(bottom=0)

    # CPU usage plot - Adjust y-axis for high CPU values
    ax3.bar(short_model_names, cpu_usage, color='salmon')
    ax3.set_title('Average CPU Usage (%)', color='white')
    ax3.set_ylabel('Percent', color='white')
    ax3.tick_params(axis='x', rotation=45, colors='white')
    ax3.tick_params(axis='y', colors='white')
    ax3.set_facecolor('#1E1E1E')

    # Set appropriate y-axis range for CPU usage
    # Get min and max values to set appropriate limits
    if cpu_usage:
        min_cpu = max(0, min(cpu_usage) * 0.9)  # 90% of minimum, but not below 0
        max_cpu = max(cpu_usage) * 1.1  # 110% of maximum
        ax3.set_ylim(bottom=min_cpu, top=max_cpu)

    # Add value labels on bars
    for i, v in enumerate(cpu_usage):
        ax3.text(i, v, f"{v:.1f}%", ha='center', va='bottom', color='white', fontweight='bold')

    plt.tight_layout()
    return fig


def main():
    st.title("📚 RAG Model Comparison System")

    # Check if Ollama is running
    if not st.session_state.ollama_checked:
        if not is_ollama_running():
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
            st.stop()
        else:
            # Get installed models
            st.session_state.installed_models = get_installed_ollama_models()
            st.session_state.ollama_checked = True

    # Sidebar: Display system specifications and recommended hardware in two columns.
    with st.sidebar:
        st.markdown("### 🖥 System Specifications")
        col_sys1, col_sys2 = st.columns(2)
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        with col_sys1:
            st.metric("CPU Cores", f"{cpu_count}")
        with col_sys2:
            st.metric("Total RAM", f"{memory.total / (1024 ** 3):.1f} GB")

        st.markdown("---")

        # Embedding model selection
        st.markdown("### 🧠 Embedding Model")
        embedding_keys = list(EMBEDDING_MODELS.keys())
        # Set "intfloat/e5-base-v2" as default
        default_emb_index = embedding_keys.index("intfloat/e5-base-v2 (Default)")
        selected_embedding = st.selectbox("Select Embedding Model", embedding_keys, index=default_emb_index)
        embedding_config = EMBEDDING_MODELS[selected_embedding]

        st.markdown(f"""
        **Model Description:**  
        {embedding_config['description']}

        **Embedding Dimensions:** {embedding_config['dimensions']}
        """)

        st.markdown("---")

        # LLM model selection
        st.markdown("### 🔧 LLM Model Configuration")
        model_keys = list(MODELS.keys())
        # Set "Neural-Chat 7B (Dialogue)" as default
        default_index = model_keys.index("Neural-Chat 7B (Dialogue)")
        selected_model = st.selectbox("Select LLM Model", model_keys, index=default_index)
        model_config = MODELS[selected_model]

        # Check if selected model is installed
        model_id = model_config["id"]
        model_installed = is_model_installed(model_id)

        st.markdown(f"""
        **Model Description:**  
        {model_config['description']}

        **Model Size:** {model_config['size']}
        """)

        if not model_installed:
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
        st.markdown("### ⚙️ Advanced Settings")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1,
                                help="Controls the randomness of responses. Lower values make output more deterministic and focused, while higher values introduce more creativity and variation.")

        k_value = st.slider("Retrieved Chunks", 1, 5, 3,
                            help="Number of document chunks to retrieve for each query. More chunks provide more context but can introduce noise.")

        # Chunking parameters
        st.markdown("### 📄 Document Processing")
        chunk_size = st.slider("Chunk Size", 256, 2048, model_config["chunk_size"], 128,
                               help="Size of text chunks in characters. Larger chunks preserve more context but may reduce retrieval precision.")

        chunk_overlap = st.slider("Chunk Overlap", 0, 200, model_config["overlap"], 10,
                                  help="Number of overlapping characters between consecutive chunks. Higher overlap helps maintain context across chunk boundaries.")

        # Button to clear cache
        if st.button("Clear Cache"):
            st.session_state.vectorstore = None
            st.session_state.embedding_model = None
            st.session_state.llm = None
            st.session_state.processed_files = []
            st.session_state.processed_pdfs = []
            st.success("Cache cleared successfully!")

        # Add information about local processing at the end of sidebar
        st.markdown("---")
        with st.expander("🔒 How This System Works Locally"):
            st.markdown("""
            This RAG system operates completely locally on your machine:

            1. **Embedding Models**: HuggingFace models run locally
            2. **LLM Models**: Ollama runs inference locally
            3. **Vector Database**: FAISS runs in-memory
            4. **Document Processing**: PDF parsing happens locally

            No external API calls are made once models are downloaded,
            making it ideal for sensitive documents.
            """)

    # Main area: PDF upload
    st.markdown("### 📚 Document Knowledge Base")

    uploaded_files = st.file_uploader("Upload up to 3 PDF files", type="pdf", accept_multiple_files=True)

    # Show warning if more than 3 files
    if uploaded_files and len(uploaded_files) > 3:
        st.warning("⚠️ For demo purposes, please limit uploads to 3 PDF files maximum.")
        uploaded_files = uploaded_files[:3]  # Take only the first 3 files

    # Display info about uploaded files
    if uploaded_files:
        # Process PDFs button
        process_button = st.button("Process PDF Files")

        if process_button:
            # Check if we need to reprocess due to embedding model change
            current_embedding = EMBEDDING_MODELS[selected_embedding]
            if st.session_state.embedding_model != current_embedding["model_name"]:
                st.session_state.vectorstore = None
                st.session_state.embedding_model = current_embedding["model_name"]
                st.session_state.processed_pdfs = []  # Reset processed PDFs list

            with st.spinner("🔄 Processing PDFs..."):
                try:
                    initial_metrics = get_process_metrics(interval=0.5)
                    start_time = time.time()

                    vectorstore, file_chunks, processed_count = process_pdfs(
                        uploaded_files,
                        chunk_size,
                        chunk_overlap,
                        current_embedding
                    )

                    end_time = time.time()
                    final_metrics = get_process_metrics(interval=0.5)
                    processing_time = end_time - start_time
                    memory_used = final_metrics["memory_mb"] - initial_metrics["memory_mb"]
                    cpu_used = final_metrics["cpu_percent"]

                    # Show stats for each file
                    if processed_count > 0:
                        st.success(f"""
                        ✅ PDF Processing Complete:
                        - Files processed: {processed_count}
                        - Processing time: {processing_time:.2f}s
                        - Memory used: {memory_used:.1f}MB
                        - CPU usage (instant): {cpu_used:.1f}%
                        - Embedding model: {current_embedding["model_name"]}
                        """)

                        # Show chunks per file in a collapsible section
                        with st.expander("Show document chunks"):
                            for filename, chunk_count in file_chunks.items():
                                file_size = round(
                                    len([f for f in uploaded_files if f.name == filename][0].getvalue()) / (
                                                1024 * 1024), 2)
                                st.markdown(f"**{filename}** ({file_size} MB): {chunk_count} chunks")

                    elif file_chunks:
                        st.info("All files were already processed. Using cached vectorstore.")
                        # Show chunks per file from previous processing
                        with st.expander("Show document chunks"):
                            for filename, chunk_count in file_chunks.items():
                                st.markdown(f"**{filename}**: {chunk_count} chunks")
                except Exception as e:
                    st.error(f"❌ Error processing PDFs: {str(e)}")
                    st.info("Try a different embedding model or check the PDF files.")
                    return
    else:
        # Info message when no files are uploaded
        st.info("""
        👆 Upload up to 3 PDF files to start. The system will process them locally and allow you to ask questions about their content.
        """)

    # Query section - only show if vectorstore exists
    if st.session_state.vectorstore is not None:
        st.markdown("---")
        st.header("🤔 Ask Questions")
        question = st.text_input("Enter your question about the documents:")

        if question:
            # Check if embedding model has changed since processing
            current_embedding = EMBEDDING_MODELS[selected_embedding]
            if st.session_state.embedding_model != current_embedding["model_name"]:
                st.warning(f"""
                ⚠️ Embedding model has changed!

                You're now using {current_embedding["model_name"]} for embeddings, but your documents 
                were processed with a different model. Please process your PDFs again with the new 
                embedding model for accurate results.
                """)
                st.session_state.vectorstore = None

            # Proceed only if vectorstore exists
            if st.session_state.vectorstore is not None:
                try:
                    with st.spinner("Generating answer..."):
                        # Measure initial consumption from the ollama process
                        get_ollama_metrics(interval=0.5)
                        start_time = time.time()

                        # Check if model has changed
                        if st.session_state.llm is None or st.session_state.current_model != selected_model:
                            st.session_state.llm = initialize_llm(model_config, temperature)
                            st.session_state.current_model = selected_model

                        llm = st.session_state.llm
                        retriever = st.session_state.vectorstore.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": k_value}
                        )

                        # Improved prompt template (English only)
                        template = """
                        Instructions: Use the provided context to answer the question clearly and directly.
                        If you cannot find the answer in the context, simply state that you cannot answer based on the available information.
                        Provide your response in a clear, professional manner. Include citations or references to the document when appropriate.

                        Context: {context}

                        Question: {question}

                        Answer:"""

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

                        # Parallel monitoring: collect samples to calculate the average usage.
                        samples = {"cpu": [], "memory": []}

                        def monitor_metrics(stop_event, samples):
                            while not stop_event.is_set():
                                current = get_ollama_metrics()
                                samples["cpu"].append(current["cpu_percent"])
                                samples["memory"].append(current["memory_mb"])
                                time.sleep(0.05)

                        result_container = {}

                        def run_chain():
                            result_container["result"] = qa_chain.invoke(question)

                        stop_event = threading.Event()
                        monitor_thread = threading.Thread(target=monitor_metrics, args=(stop_event, samples))
                        chain_thread = threading.Thread(target=run_chain)

                        monitor_thread.start()
                        chain_thread.start()

                        # Update the UI placeholder with readings and status
                        placeholder = st.empty()
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # Animation for progress indicator
                        progress_value = 0
                        progress_increment = 0.015  # Adjust for slower/faster animation
                        progress_phases = ["Retrieving context", "Processing information", "Generating response",
                                           "Finalizing answer"]
                        phase_index = 0

                        while chain_thread.is_alive():
                            current = get_ollama_metrics(interval=0.2)

                            # Update metrics display
                            placeholder.markdown(
                                f"**Current CPU Usage:** {current['cpu_percent']:.1f}% | **Current Memory Usage:** {current['memory_mb']:.1f} MB")

                            # Update progress animation
                            progress_value = min(0.95, progress_value + progress_increment)
                            progress_bar.progress(progress_value)

                            # Update status text with changing messages
                            if progress_value > (phase_index + 1) * 0.23 and phase_index < len(progress_phases) - 1:
                                phase_index += 1

                            status_text.markdown(f"**Status:** {progress_phases[phase_index]}...")

                            time.sleep(0.1)

                        # Complete the progress bar
                        progress_bar.progress(1.0)
                        status_text.markdown("**Status:** Answer ready!")
                        time.sleep(0.5)

                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()

                        chain_thread.join()
                        stop_event.set()
                        monitor_thread.join()

                        end_time = time.time()
                        response_time = end_time - start_time

                        # Calculate the average CPU and memory from the collected samples
                        avg_cpu = sum(samples["cpu"]) / len(samples["cpu"]) if samples["cpu"] else 0.0
                        avg_memory = sum(samples["memory"]) / len(samples["memory"]) if samples["memory"] else 0.0

                        response = result_container["result"]

                        # Save metrics with embedding model information
                        save_metrics(
                            selected_model,
                            EMBEDDING_MODELS[selected_embedding]["model_name"],
                            response_time,
                            len(response['result'].split()),
                            avg_memory,
                            avg_cpu,
                            question
                        )

                        # Show the response in a more elegant style
                        st.markdown("### 📝 Answer:")
                        st.markdown(response['result'])

                        # Show only the most relevant source document
                        if response.get('source_documents') and len(response['source_documents']) > 0:
                            most_relevant_doc = response['source_documents'][0]  # First doc is most relevant
                            max_preview_length = 200  # Character limit for preview

                            # Prepare preview text (shortened)
                            doc_text = most_relevant_doc.page_content
                            if len(doc_text) > max_preview_length:
                                preview_text = doc_text[:max_preview_length] + "..."
                            else:
                                preview_text = doc_text

                            with st.expander("View Source Document"):
                                st.markdown("**Most Relevant Source:**")

                                # Show source file name if available
                                if hasattr(most_relevant_doc,
                                           'metadata') and most_relevant_doc.metadata and 'source' in most_relevant_doc.metadata:
                                    source_file = most_relevant_doc.metadata['source']
                                    st.markdown(f"📄 **File:** {source_file}")

                                st.markdown(f"```\n{preview_text}\n```")

                                if hasattr(most_relevant_doc, 'metadata') and most_relevant_doc.metadata:
                                    st.markdown("**Metadata:**")
                                    for key, value in most_relevant_doc.metadata.items():
                                        if key != 'source':  # Already displayed source above
                                            st.markdown(f"- **{key}**: {value}")

                        st.markdown("### 📊 Performance Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Response Time", f"{response_time:.2f}s")
                        with col2:
                            st.metric("Avg Memory Used", f"{avg_memory:.1f}MB")
                        with col3:
                            st.metric("Avg CPU Usage", f"{avg_cpu:.1f}%")
                        with col4:
                            st.metric("Output Tokens", len(response['result'].split()))

                        if len(st.session_state.model_metrics) > 1:
                            st.markdown("### 📈 Model Comparison")

                            # Comparison table
                            comparison_data = []
                            for model_key, metrics in st.session_state.model_metrics.items():
                                # Get the latest query for this model combination
                                latest_query = metrics[-1]['query']
                                # Truncate query if too long
                                if len(latest_query) > 40:
                                    display_query = latest_query[:37] + "..."
                                else:
                                    display_query = latest_query

                                # Split information from the first metric entry
                                llm_model_name = metrics[0]['llm_model']
                                embedding_model = metrics[0]['embedding_model'].split('/')[
                                    -1]  # Just the model name without path

                                avg_metrics = {
                                    "LLM Model": llm_model_name,
                                    "Embedding Model": embedding_model,
                                    "Latest Query": display_query,
                                    "Avg Response Time": f"{sum(m['response_time'] for m in metrics) / len(metrics):.2f}s",
                                    "Avg Memory Used": f"{sum(m['memory_used'] for m in metrics) / len(metrics):.1f}MB",
                                    "Avg CPU Usage": f"{sum(m['cpu_used'] for m in metrics) / len(metrics):.1f}%"
                                }
                                comparison_data.append(avg_metrics)
                            st.table(comparison_data)

                            # Visualization
                            fig = plot_comparison_charts()
                            if fig:
                                st.pyplot(fig)

                except Exception as e:
                    st.error(f"❌ Error generating answer: {str(e)}")
                    st.info("Try adjusting model parameters or using a different model.")
            else:
                st.error("Please process the documents with the current embedding model first.")


if __name__ == "__main__":
    main()
