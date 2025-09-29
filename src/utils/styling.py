"""
Styling Utility
==============

Provides custom CSS styling for the Streamlit application to create
a professional and modern user interface.
"""

import streamlit as st

def apply_custom_styling():
    """Apply custom CSS styling to the Streamlit application"""
    
    custom_css = """
    <style>
        /* Main header styling */
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
            padding: 1rem 0;
            background: linear-gradient(90deg, #f0f8ff 0%, #e6f3ff 100%);
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Metric container styling */
        .metric-container {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            border-left: 4px solid #1f77b4;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        /* Insight box styling */
        .insight-box {
            background-color: #e8f4fd;
            border-left: 5px solid #1f77b4;
            padding: 1.5rem;
            margin: 1rem 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        /* Warning box styling */
        .warning-box {
            background-color: #fff3cd;
            border-left: 5px solid #ffc107;
            padding: 1.5rem;
            margin: 1rem 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        /* Success box styling */
        .success-box {
            background-color: #d4edda;
            border-left: 5px solid #28a745;
            padding: 1.5rem;
            margin: 1rem 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        /* Database configuration styling */
        .database-config {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        /* Investigator card styling */
        .investigator-card {
            background-color: #ffffff;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: transform 0.2s ease-in-out;
        }
        
        .investigator-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* Rank styling */
        .rank-1 {
            background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%);
            border-left: 5px solid #ffc107;
        }
        
        .rank-2 {
            background: linear-gradient(135deg, #c0c0c0 0%, #e9ecef 100%);
            border-left: 5px solid #6c757d;
        }
        
        .rank-3 {
            background: linear-gradient(135deg, #cd7f32 0%, #d4a574 100%);
            border-left: 5px solid #8b4513;
        }
        
        .rank-other {
            background-color: #f8f9fa;
            border-left: 5px solid #6c757d;
        }
        
        /* Header line styling */
        .header-line {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 2rem;
            padding: 1rem;
            background-color: #f8f9fa;
            border-radius: 8px;
        }
        
        .subheader {
            font-size: 1.2rem;
            font-weight: bold;
            color: #495057;
        }
        
        .description {
            font-size: 1rem;
            color: #6c757d;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f8f9fa;
            border-radius: 8px 8px 0px 0px;
            gap: 1rem;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #1f77b4;
            color: white;
        }
        
        /* Button styling */
        .stButton > button {
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.2s ease-in-out;
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* Dataframe styling */
        .dataframe {
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #f8f9fa;
        }
        
        /* Metric styling */
        .metric-container {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 10px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            border: 1px solid #dee2e6;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        /* Chart container styling */
        .chart-container {
            background-color: white;
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border: 1px solid #e9ecef;
        }
        
        /* Progress bar styling */
        .stProgress > div > div > div > div {
            background-color: #1f77b4;
        }
        
        /* Selectbox styling */
        .stSelectbox > div > div {
            border-radius: 8px;
        }
        
        /* Number input styling */
        .stNumberInput > div > div > input {
            border-radius: 8px;
        }
        
        /* Text input styling */
        .stTextInput > div > div > input {
            border-radius: 8px;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #f8f9fa;
            border-radius: 8px;
            font-weight: 500;
        }
        
        /* Footer styling */
        .footer {
            text-align: center;
            color: #6c757d;
            font-size: 0.9em;
            padding: 2rem 0;
            border-top: 1px solid #e9ecef;
            margin-top: 2rem;
        }
    </style>
    """
    menu_style()
    st.markdown(custom_css, unsafe_allow_html=True)
    


def menu_style():
    """
    Apply custom styling to the Streamlit application
    """
    custom_css = """
    <style>
    div[data-testid="stVerticalBlock"] {
        gap:0.5rem !important; /* Remove gap between columns */  
    }
    .rc-overflow {
        background-color: #005A9C ; /* A light green shade */
        padding: 25px; /* Add some padding around the content */
        padding-top: 20px; /* Add some padding at the top */
        padding-bottom: 20px; /* Add some padding at the bottom */
        padding-left: 40px; /* Add some padding at the left */
        border-radius: 5px; /* Slightly rounded corners */
    }
    .rc-overflow-item >div>div>span {
        font-size: 1.2rem !important; /* Slightly larger font size */
        color: white !important;
    }
    [data-testid="stMainBlockContainer"] {
            padding-top: 5rem !important; /* Adjust the top padding of the main content */
            padding-left: 3rem !important; /* Adjust the left padding of the main content */
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def create_metric_card(title: str, value: str, delta: str = None, help_text: str = None):
    """
    Create a styled metric card
    
    Args:
        title: The metric title
        value: The metric value
        delta: The delta/change value (optional)
        help_text: Help text for the metric (optional)
    """
    delta_html = f"<div style='font-size: 0.8em; color: #28a745;'>{delta}</div>" if delta else ""
    help_html = f"<div style='font-size: 0.7em; color: #6c757d; margin-top: 0.5rem;'>{help_text}</div>" if help_text else ""
    
    card_html = f"""
    <div class="metric-container">
        <div style='font-size: 0.9em; color: #6c757d; margin-bottom: 0.5rem;'>{title}</div>
        <div style='font-size: 1.5rem; font-weight: bold; color: #1f77b4;'>{value}</div>
        {delta_html}
        {help_html}
    </div>
    """
    
    return card_html

def create_insight_box(title: str, content: str, box_type: str = "info"):
    """
    Create a styled insight box
    
    Args:
        title: The box title
        content: The box content
        box_type: Type of box ("info", "warning", "success")
    """
    box_class = {
        "info": "insight-box",
        "warning": "warning-box", 
        "success": "success-box"
    }.get(box_type, "insight-box")
    
    box_html = f"""
    <div class="{box_class}">
        <h4 style='margin-top: 0; color: #495057;'>{title}</h4>
        <div style='color: #6c757d;'>{content}</div>
    </div>
    """
    
    return box_html 