"""
Tracks and manages performance metrics for model comparison.
"""
import time
import psutil
import datetime
import threading
import streamlit as st
import matplotlib.pyplot as plt


class MetricsTracker:
    """
    Tracks and stores performance metrics for different models.
    """

    def __init__(self):
        """
        Initialize the metrics tracker.
        """
        self.metrics = {}

    def get_process_metrics(self, interval=0.1):
        """
        Get CPU and memory metrics for the current process.

        Args:
            interval (float): Interval for CPU measurement

        Returns:
            dict: Dictionary with memory_mb and cpu_percent keys
        """
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        cpu_percent = process.cpu_percent(interval=interval)
        return {
            "memory_mb": memory_mb,
            "cpu_percent": cpu_percent
        }

    def get_ollama_metrics(self, interval=0.1):
        """
        Get CPU and memory metrics for Ollama processes.

        Args:
            interval (float): Interval for CPU measurement

        Returns:
            dict: Dictionary with memory_mb and cpu_percent keys
        """
        matching = []
        for proc in psutil.process_iter(['name']):
            try:
                if "ollama" in proc.name().lower():
                    matching.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                print(f"Error accessing process: {e}")
                continue

        # Reset CPU counters for each matching process
        for p in matching:
            try:
                p.cpu_percent()
            except Exception as e:
                print(f"Error resetting CPU counter: {e}")
                pass

        time.sleep(interval)

        cpu_sum = 0.0
        mem_sum = 0.0
        for p in matching:
            try:
                cpu_val = p.cpu_percent()
                cpu_sum += cpu_val
            except Exception as e:
                print(f"Error reading CPU: {e}")

            try:
                mem = p.memory_info().rss
                mem_sum += mem / (1024 * 1024)
            except Exception as e:
                print(f"Error reading memory: {e}")

        return {
            "cpu_percent": cpu_sum,
            "memory_mb": mem_sum
        }

    def start_monitoring(self, callback=None):
        """
        Start monitoring system metrics.

        Args:
            callback: Function to call with metrics

        Returns:
            tuple: (stop_event, samples_dict) - Objects to control monitoring
        """
        samples = {"cpu": [], "memory": []}
        stop_event = threading.Event()

        # Store samples in session state for thread safety
        if 'metrics_samples' not in st.session_state:
            st.session_state.metrics_samples = samples

        def monitor_metrics():
            while not stop_event.is_set():
                current = self.get_ollama_metrics()

                # Update directly in the session state
                if 'metrics_samples' in st.session_state:
                    if 'cpu' not in st.session_state.metrics_samples:
                        st.session_state.metrics_samples['cpu'] = []
                    if 'memory' not in st.session_state.metrics_samples:
                        st.session_state.metrics_samples['memory'] = []

                    st.session_state.metrics_samples['cpu'].append(current["cpu_percent"])
                    st.session_state.metrics_samples['memory'].append(current["memory_mb"])

                # Don't try to update UI from this thread
                # if callback:
                #     callback(current)

                # Also update the local samples dictionary as a backup
                samples["cpu"].append(current["cpu_percent"])
                samples["memory"].append(current["memory_mb"])

                time.sleep(0.05)

        # Start the monitoring thread
        monitor_thread = threading.Thread(target=monitor_metrics)
        monitor_thread.daemon = True
        monitor_thread.start()

        return stop_event, samples

    def save_metrics(self, model_name, embedding_model_name, response_time, tokens, memory_used, cpu_used, query):
        """
        Save performance metrics for a query.

        Args:
            model_name (str): Name of the LLM model
            embedding_model_name (str): Name of the embedding model
            response_time (float): Response time in seconds
            tokens (int): Number of tokens in the response
            memory_used (float): Memory used in MB
            cpu_used (float): CPU usage percentage
            query (str): The query that was asked

        Returns:
            str: The key used to store the metrics
        """
        # Create a unique key for this model combination
        model_key = f"{model_name} + {embedding_model_name.split('/')[-1]}"

        if model_key not in self.metrics:
            self.metrics[model_key] = []

        # Save the metrics
        self.metrics[model_key].append({
            'timestamp': datetime.datetime.now().strftime("%H:%M:%S"),
            'llm_model': model_name,
            'embedding_model': embedding_model_name,
            'response_time': response_time,
            'tokens': tokens,
            'memory_used': memory_used,
            'cpu_used': cpu_used,
            'query': query
        })

        return model_key

    def get_all_metrics(self):
        """
        Get all metrics stored.

        Returns:
            dict: All metrics
        """
        return self.metrics

    def get_model_keys(self):
        """
        Get list of all model combination keys.

        Returns:
            list: List of model keys
        """
        return list(self.metrics.keys())

    def get_model_metrics(self, model_key):
        """
        Get metrics for a specific model combination.

        Args:
            model_key (str): The model combination key

        Returns:
            list: List of metric dictionaries
        """
        return self.metrics.get(model_key, [])

    def get_average_metrics(self, model_key):
        """
        Get average metrics for a specific model combination.

        Args:
            model_key (str): The model combination key

        Returns:
            dict: Dictionary with average metrics
        """
        metrics = self.get_model_metrics(model_key)
        if not metrics:
            return {}

        avg_response_time = sum(m['response_time'] for m in metrics) / len(metrics)
        avg_memory_used = sum(m['memory_used'] for m in metrics) / len(metrics)
        avg_cpu_used = sum(m['cpu_used'] for m in metrics) / len(metrics)

        # Get the LLM and embedding model names from the first entry
        llm_model = metrics[0]['llm_model']
        embedding_model = metrics[0]['embedding_model'].split('/')[-1]

        # Get the latest query
        latest_query = metrics[-1]['query']

        return {
            "LLM Model": llm_model,
            "Embedding Model": embedding_model,
            "Latest Query": latest_query if len(latest_query) <= 40 else latest_query[:37] + "...",
            "Avg Response Time": f"{avg_response_time:.2f}s",
            "Avg Memory Used": f"{avg_memory_used:.1f}MB",
            "Avg CPU Usage": f"{avg_cpu_used:.1f}%"
        }

    def plot_comparison_charts(self):
        """
        Generate comparison charts for models.

        Returns:
            matplotlib.figure.Figure: Figure with comparison charts
        """
        if not self.metrics:
            return None

        # Prepare data for plotting
        model_keys = self.get_model_keys()
        resp_times = []
        mem_usage = []
        cpu_usage = []

        for model_key in model_keys:
            metrics = self.get_model_metrics(model_key)
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
        ax1.set_ylim(bottom=0)

        # Memory usage plot
        ax2.bar(short_model_names, mem_usage, color='lightgreen')
        ax2.set_title('Average Memory Usage (MB)', color='white')
        ax2.set_ylabel('MB', color='white')
        ax2.tick_params(axis='x', rotation=45, colors='white')
        ax2.tick_params(axis='y', colors='white')
        ax2.set_facecolor('#1E1E1E')
        ax2.set_ylim(bottom=0)

        # CPU usage plot
        ax3.bar(short_model_names, cpu_usage, color='salmon')
        ax3.set_title('Average CPU Usage (%)', color='white')
        ax3.set_ylabel('Percent', color='white')
        ax3.tick_params(axis='x', rotation=45, colors='white')
        ax3.tick_params(axis='y', colors='white')
        ax3.set_facecolor('#1E1E1E')

        # Set appropriate y-axis range for CPU usage
        if cpu_usage:
            min_cpu = max(0, min(cpu_usage) * 0.9)
            max_cpu = max(cpu_usage) * 1.1
            ax3.set_ylim(bottom=min_cpu, top=max_cpu)

        # Add value labels on bars
        for i, v in enumerate(cpu_usage):
            ax3.text(i, v, f"{v:.1f}%", ha='center', va='bottom', color='white', fontweight='bold')

        plt.tight_layout()
        return fig

    def clear(self):
        """
        Clear all stored metrics.
        """
        self.metrics = {}
