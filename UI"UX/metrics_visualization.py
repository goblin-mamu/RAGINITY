"""
Visualization components for performance metrics.
"""
import matplotlib.pyplot as plt
import streamlit as st


class MetricsVisualization:
    """
    Handles visualization of performance metrics.
    """

    @staticmethod
    def create_comparison_table(metrics_tracker):
        """
        Create a comparison table from metrics data.

        Args:
            metrics_tracker: MetricsTracker instance with data

        Returns:
            list: List of dictionaries for table display
        """
        if not metrics_tracker:
            return []

        comparison_data = []
        for model_key in metrics_tracker.get_model_keys():
            avg_metrics = metrics_tracker.get_average_metrics(model_key)
            comparison_data.append(avg_metrics)

        return comparison_data

    @staticmethod
    def display_realtime_metrics(metrics):
        """
        Display real-time metrics during generation.

        Args:
            metrics (dict): Dictionary with memory_mb and cpu_percent keys
        """
        placeholder = st.empty()
        placeholder.markdown(
            f"**Current CPU Usage:** {metrics['cpu_percent']:.1f}% | "
            f"**Current Memory Usage:** {metrics['memory_mb']:.1f} MB"
        )
        return placeholder

    @staticmethod
    def display_performance_metrics(metrics):
        """
        Display performance metrics after generation.

        Args:
            metrics (dict): Dictionary with performance metrics
        """
        st.markdown("### 📊 Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Response Time", f"{metrics['response_time']:.2f}s")
        with col2:
            st.metric("Avg Memory Used", f"{metrics['memory_used']:.1f}MB")
        with col3:
            st.metric("Avg CPU Usage", f"{metrics['cpu_used']:.1f}%")
        with col4:
            st.metric("Output Tokens", metrics['tokens'])

    @staticmethod
    def display_comparison_table(comparison_data):
        """
        Display the model comparison table.

        Args:
            comparison_data (list): List of dictionaries with metrics
        """
        if not comparison_data:
            return

        st.markdown("### 📈 Model Comparison")
        st.table(comparison_data)

    @staticmethod
    def create_comparison_charts(metrics_tracker):
        """
        Create comparison charts for models.

        Args:
            metrics_tracker: MetricsTracker instance

        Returns:
            matplotlib.figure.Figure: Figure with comparison charts
        """
        return metrics_tracker.plot_comparison_charts()
