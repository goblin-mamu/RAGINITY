"""
Manages custom CSS styles for the application.
"""
import streamlit as st


class StyleManager:
    """
    Manages custom CSS styling for the application UI.
    """

    @staticmethod
    def apply_styles():
        """
        Apply custom CSS styling to the Streamlit application.
        """
        st.markdown(
            """
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
            """,
            unsafe_allow_html=True
        )
