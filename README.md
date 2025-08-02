# RAGINITY
#🧠 RAG Model Comparison System
Compare, Benchmark, and Explore RAG Pipelines Locally.
This is a fully offline demo application that allows you to compare various Retrieval-Augmented Generation (RAG) pipelines using Ollama for local LLM inference and Hugging Face embeddings for document indexing.

#🌟 Key Features
📄 PDF Processing: Upload and analyze PDFs for intelligent document-level QA.

🔎 Custom Embeddings: Choose from multiple Hugging Face embedding models.

🤖 Model Flexibility: Swap between various LLMs via Ollama for response generation.

📊 Live Metrics: Measure response time, memory, and CPU usage per configuration.

📈 Visual Comparison: Real-time charts to compare performance across models.

🖥️ Fully Local: No external APIs—everything runs locally on your machine.

#🧠 Available Embedding Models
Model	Description	Dimensions
intfloat/e5-base-v2	Balanced performance for general text search	768
BAAI/bge-m3	Multilingual, high-ranking for global queries	1024
mixedbread-ai/mxbai-embed-large-v1	Excellent precision for retrieval	768
nomic-ai/nomic-embed-text-v1.5	Designed for document-level semantic search	768
Snowflake/arctic-embed-s	Lightweight and efficient for fast iterations	384
sentence-transformers/all-MiniLM-L6-v2	Compact, fast, great for small-scale tasks	384

#🤖 Available LLMs via Ollama
Model	Size	Context	Strength
Neural-Chat 7B	4.1 GB	8192	Conversational and friendly
DeepSeek-R1 1.5B	1.1 GB	8192	Very fast and resource-light
DeepSeek-R1 7B	4.7 GB	8192	Logical reasoning and QA
Mistral 7B	4.1 GB	8192	Balanced and reliable
Llama3 8B	4.7 GB	4096	Great for broad understanding
Phi3.5 3.8B	2.2 GB	4096	Small but surprisingly strong
Gemma2 9B	5.2 GB	8192	High-quality model from Google
DeepSeek-R1 8B	4.8 GB	8192	Advanced reasoning
Qwen2.5 7B	4.1 GB	32768	Multilingual + long-context
OpenCoder 8B	4.7 GB	16384	Built for code and tech docs

#🖥️ System Requirements
Component	Requirement
Python	3.9+
CPU	4+ cores (8+ recommended)
RAM	8 GB minimum (16 GB+ recommended)
Disk	10 GB+ (for models)
OS	Windows, macOS, or Linux

🚀 Installation Guide
1️⃣ Install Ollama (Local LLM Runtime)
Windows:
Download for Windows

macOS / Linux:


curl -fsSL https://ollama.com/install.sh | sh
2️⃣ Set Up Python Environment
bash
Copy
Edit
# Clone the repo
git clone https://github.com/your-username/rag-model-comparison.git
cd rag-model-comparison

# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
✅ requirements.txt Content
shell
Copy
Edit
streamlit>=1.24.0
langchain>=0.1.0
langchain-community>=0.0.10
langchain-huggingface>=0.0.6
langchain-ollama>=0.0.3
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2
pdfplumber>=0.10.2
numpy==1.24.3
torch>=2.0.0
protobuf>=4.25.1
scikit-learn==1.2.2
transformers>=4.30.0
psutil>=5.9.0
watchdog==6.0.0
matplotlib>=3.5.0
pandas>=1.3.0
3️⃣ Download Ollama Models (Optional but Recommended)
bash
Copy
Edit
# Base models
ollama pull neural-chat:7b
ollama pull deepseek-r1:1.5b
ollama pull mistral:7b
ollama pull llama3:8b

# Additional models
ollama pull phi3.5:3.8b
ollama pull gemma2:9b
ollama pull qwen2.5:7b
ollama pull opencoder:8b
To list all models:

bash
Copy
Edit
ollama list
#💡 How to Use
Start Ollama in the background.

Activate the Python environment:

bash
Copy
Edit
venv\Scripts\activate   # Windows
source venv/bin/activate  # macOS/Linux
Run the app:

bash
Copy
Edit
streamlit run app.py
Go to http://localhost:8501 in your browser.

#🧪 Using the App
Select Models: Choose embedding + LLM models from the sidebar.

Upload Documents: Upload any PDF via the UI.

Set Parameters: Adjust temperature, chunk size, top-k retrieval, etc.

Ask Questions: Enter questions about the document.

Visual Comparison: Check charts to analyze response times, resource usage, and output quality.

#🛠️ Troubleshooting
Ollama Issues
Not starting? Check if Ollama service is running.

Model download failed? Try manual pull or free disk space.

Slow inference? Use smaller models like phi3.5 or deepseek-r1:1.5b.

Embedding Issues
CUDA errors? Ensure models are set to run on CPU.

Out of memory? Use smaller embedding models (arctic-embed-s, MiniLM).

Other
Long initial load? Models are downloaded on first use.

Chunk error? Reduce chunk size in advanced settings.

#🤝 Contributing
Want to help?


# Fork this repo
# Create a feature branch
git checkout -b feature/your-feature

# Make changes and commit
git commit -m "Added cool feature"

# Push and open PR
git push origin feature/your-feature
