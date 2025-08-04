"""
Configuration for LLM and embedding models used in the RAG comparison system.
"""

# Model configuration with recommended hardware
LLM_MODELS = {
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

# Default RAG prompt template
DEFAULT_RAG_PROMPT = """
Instructions: Use the provided context to answer the question clearly and directly.
If you cannot find the answer in the context, simply state that you cannot answer based on the available information.
Provide your response in a clear, professional manner. Include citations or references to the document when appropriate.

Context: {context}

Question: {question}

Answer:
"""