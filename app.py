"""
🔬 Clinical Trial Analytics Dashboard - Main Application
=======================================================

A modern, modular Streamlit application for clinical trial data analysis.
This is the main entry point that provides navigation to different analysis modules.

Author: AI Assistant
Created: 2024
"""

import streamlit as st
from datetime import datetime
import sys
import os


# Add the src directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.database.connection_manager import DatabaseConnectionManager
from src.utils.config_loader import ConfigLoader
from src.utils.styling import apply_custom_styling
# from src.Module.home import render_home_page
from src.utils.chat_boat import render_chat_boat

# Streamlit Float for floating elements
from streamlit_float import *

# Page configuration
st.set_page_config(
    page_title="Clinical Trial Analytics Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom styling
apply_custom_styling()

def render_database_connection_ui():
    """Render database connection configuration in sidebar"""
    
    st.sidebar.markdown("## 🗄️ Database Connection")
    
    # Database type selection
    db_type = st.sidebar.selectbox(
        "Select Database Type",
        ["Sample Database", "Real AACT Database", "Local Database"],
        help="Choose between sample data, real AACT database, or connect to a local database"
    )
    
    if db_type == "Sample Database":
        st.sidebar.markdown("""
        **📊 Sample Database**
        - Quick start with synthetic data
        - No credentials required
        - 1,000 sample trials
        """)
        
        if st.sidebar.button("🚀 Load Sample Database", type="primary"):
            with st.spinner("Creating sample database..."):
                success = st.session_state.db_manager.create_sample_database()
                if success:
                    st.sidebar.success("✅ Sample database loaded!")
                    st.rerun()
                else:
                    st.sidebar.error("❌ Failed to create sample database")
    
    elif db_type == "Real AACT Database":
        st.sidebar.markdown("""
        **🌐 Real AACT Database**
        - Live clinical trial data
        - Requires free registration
        - 400,000+ real trials
        """)
        
        st.sidebar.markdown("**📋 Connection Details**")
        
        # Connection form
        with st.sidebar.form("aact_connection"):
            host = st.text_input("Host", value="aact-db.ctti-clinicaltrials.org", help="AACT database host")
            port = st.number_input("Port", value=5432, min_value=1, max_value=65535, help="Database port")
            database = st.text_input("Database", value="aact", help="Database name")
            username = st.text_input("Username", help="Your AACT username")
            password = st.text_input("Password", type="password", help="Your AACT password")
            
            submitted = st.form_submit_button("🔗 Connect to AACT", type="primary")
            
            if submitted:
                if username and password:
                    with st.spinner("Connecting to AACT database..."):
                        success = st.session_state.db_manager.connect_to_real_aact(
                            host, port, database, username, password
                        )
                        if success:
                            st.sidebar.success("✅ Connected to AACT database!")
                            st.rerun()
                        else:
                            st.sidebar.error("❌ Failed to connect to AACT database")
                else:
                    st.sidebar.error("❌ Please provide username and password")
    
    elif db_type == "Local Database":
        st.sidebar.markdown("""
        **💾 Local Database**
        - Connect to your own local Postgres database
        - Use your own credentials and host information
        """)
        st.sidebar.markdown("**📋 Local Database Connection Details**")
        with st.sidebar.form("local_db_connection"):
            host = st.text_input("Host", value="localhost", help="Local database host")
            port = st.number_input("Port", value=5432, min_value=1, max_value=65535, help="Database port")
            database = st.text_input("Database", value="aact", help="Database name")
            username = st.text_input("Username", help="Your local database username")
            password = st.text_input("Password", type="password", help="Your local database password")
            
            submitted = st.form_submit_button("🔗 Connect to Local Database", type="primary")
            
            if submitted:
                if username and password:
                    with st.spinner("Connecting to local database..."):
                        success = st.session_state.db_manager.connect_to_local_database(
                            host, port, database, username, password
                        )
                        if success:
                            st.sidebar.success("✅ Connected to local database!")
                            st.rerun()
                        else:
                            st.sidebar.error("❌ Failed to connect to local database")
                else:
                    st.sidebar.error("❌ Please provide username and password")
        
        # Help information
        with st.sidebar.expander("ℹ️ How to get AACT credentials"):
            st.markdown("""
            1. **Register for free** at [aact.ctti-clinicaltrials.org](https://aact.ctti-clinicaltrials.org)
            2. **Complete registration** and wait for approval email
            3. **Get credentials** from your account dashboard
            4. **Use credentials** in the form above
            
            **Note**: Registration is free and typically approved within 24 hours.
            """)
    
    # Current connection status
    if hasattr(st.session_state, 'database_loaded') and st.session_state.database_loaded:
        st.sidebar.markdown("---")
        if st.session_state.database_type == 'real':
            st.sidebar.success("✅ Connected to Real AACT Database")
        elif st.session_state.database_type == 'local':
            st.sidebar.success("✅ Connected to Local Database")
        else:
            st.sidebar.success("✅ Sample Database Loaded")

        
        # Disconnect option
        if st.sidebar.button("🔌 Disconnect"):
            if 'database_loaded' in st.session_state:
                del st.session_state.database_loaded
            if 'database_type' in st.session_state:
                del st.session_state.database_type
            st.session_state.db_manager = DatabaseConnectionManager()
            st.sidebar.success("🔌 Disconnected")
            st.rerun()

def main():
    """Main application entry point"""
    
    # Initialize session state
    if 'config' not in st.session_state:
        st.session_state.config = ConfigLoader()
    
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseConnectionManager()
    
    st.sidebar.title("🔬 Clinical Trial Analytics Dashboard")
    # Render database connection UI in sidebar
    render_database_connection_ui()

    
    
    # # Header
    # st.markdown("""
    # <div class="main-header">
    #     🔬 Clinical Trial Analytics Dashboard
    # </div>
    # """, unsafe_allow_html=True)

    pages = {
        "🏠 Dashboard": [
            st.Page("src/Module/home.py", title="Overview & Features", icon="🎯")
        ],
        
        "📊 Analytics & Insights": [
            st.Page("src/Module/trial_landscape.py", title="Trial Landscape Analysis", icon="📈"),           
            st.Page("src/Module/success_analytics.py", title="Success Metrics", icon="✅"),
            st.Page("src/Module/innovation_insights.py", title="Innovation Trends", icon="💡"),
            st.Page("src/Module/predictive_analytics.py", title="Predictive Analytics", icon="🤖"),
        ],

        "🎯 Trial Planning": [
            st.Page("src/Module/trail_feasibility.py", title="Site Feasibility", icon="🔍"),
            # st.Page("src/Module/access_equity.py", title="Patient Access Analysis", icon="🌍"),
            st.Page("src/Module/facility_ranking.py", title="Facility Ranking", icon="🏥"),
            # st.Page("src/Module/investigator_ranking.py", title="Investigator Assessment", icon="👨‍⚕️")
        ],

        "📑 Reports": [
            st.Page("src/Module/executive_summary.py", title="Executive Summary", icon="📊"),
            st.Page("src/Module/study_reports.py", title="Clinical Study Reports", icon="📄"), 
            st.Page("src/Module/sponsor_profiles.py", title="Sponsor Analysis", icon="🏢")
        ]
    }
    # all_pages=[st.Page("src/Module/home.py",title="Home",icon="🏠"),Module]
    
    pg = st.navigation(pages,position="top", expanded=False)
    # st.write(pg.url_path)
    pg.run()
    # Render the home page content
    # render_home_page()

    render_chat_boat()
    
    # Footer
    st.markdown("---")
    render_footer()

def render_footer():
    """Render application footer"""
    db_status = "📊 Sample Database"
    if hasattr(st.session_state, 'database_loaded') and st.session_state.database_loaded:
        if st.session_state.database_type == 'real':
            db_status = "🌐 Real AACT Database"
        elif st.session_state.database_type == 'local':
            db_status = "💾 Local Database"
        else:
            db_status = "📊 Sample Database"
    
    footer_html = f"""
    <div style='text-align: center; color: #666; font-size: 0.9em; padding: 1rem 0;'>
        <p>🔬 Clinical Trial Analytics Dashboard | Powered by AACT Database | Built with Streamlit</p>
        <p>📊 Advanced Analytics for Evidence-Based Decision Making | {db_status}</p>
        <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 