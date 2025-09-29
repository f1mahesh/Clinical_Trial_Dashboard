"""
Home Page
=========

Landing page for the Clinical Trial Analytics Dashboard.
Provides database connection options and overview of capabilities.
"""

import streamlit as st
from src.utils.styling import create_insight_box

def render_home_page():
    """Render the home page"""
    
    st.header("🏠 Welcome to Clinical Trial Analytics Dashboard")
    
    # Check if database is loaded
    if not hasattr(st.session_state, 'database_loaded') or not st.session_state.database_loaded:
        render_database_setup()
    else:
        render_dashboard_overview()

def render_database_setup():
    """Render database connection setup"""
    
    st.markdown("""
    <div class="insight-box">
        <h4>🔬 Comprehensive Analytics Platform</h4>
        <p>Explore innovative insights from clinical trial data using the AACT database. 
        This dashboard provides advanced analytics for researchers, policymakers, and industry professionals.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.warning("Please connect to a database from the sidebar to start exploring analytics.")
    
    # Show available options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Sample Database
        - **Quick Start**: Ready in seconds
        - **No Setup**: No credentials needed
        - **Demo Data**: 1,000 synthetic trials
        - **Full Features**: All analytics available
        
        Perfect for testing and exploring dashboard capabilities.
        """)
        
        if st.button("🚀 Load Sample Database", type="primary"):
            with st.spinner("Creating sample database..."):
                success = st.session_state.db_manager.create_sample_database()
                if success:
                    st.success("✅ Sample database loaded successfully!")
                    st.rerun()
    
    with col2:
        st.markdown("""
        ### 🌐 Real AACT Database
        - **Live Data**: 400,000+ real clinical trials
        - **Current**: Updated monthly from ClinicalTrials.gov
        - **Comprehensive**: Complete trial information
        - **Research Grade**: Publication-quality data
        
        Requires free registration at [aact.ctti-clinicaltrials.org](https://aact.ctti-clinicaltrials.org)
        """)
        
        st.info("💡 Use the sidebar to configure real AACT database connection")
    
    # Features overview
    st.markdown("---")
    st.subheader("🎯 Dashboard Features")
    
    features_col1, features_col2 = st.columns(2)
    
    with features_col1:
        st.markdown("""
        **📊 Trial Landscape Overview**
        - Study volume and distribution analysis
        - Phase and status breakdowns
        - Geographic distribution insights
        - Sponsor analysis and trends
        
        **✅ Success & Completion Analytics**
        - Completion rates by phase
        - Results reporting compliance
        - Duration analysis and trends
        - Success factor identification
        """)
    
    with features_col2:
        st.markdown("""
        **💡 Innovation & Market Insights**
        - Intervention type trends
        - Emerging therapeutic areas
        - Sponsor innovation patterns
        - Market opportunity analysis
        
        **🌍 Access & Equity Analysis**
        - Geographic access patterns
        - Gender and age inclusion
        - Diversity metrics
        - Equity considerations
        """)
    
    # Advanced features
    st.markdown("---")
    st.subheader("🚀 Advanced Analytics")
    
    advanced_col1, advanced_col2, advanced_col3 = st.columns(3)
    
    with advanced_col1:
        st.markdown("""
        **👨‍⚕️ Investigator Ranking**
        - ML-powered performance analysis
        - Multi-dimensional scoring
        - Collaboration patterns
        - Success prediction
        """)
    
    with advanced_col2:
        st.markdown("""
        **🏥 Facility Ranking**
        - Performance benchmarking
        - Efficiency metrics
        - Innovation scoring
        - Capacity analysis
        """)
    
    with advanced_col3:
        st.markdown("""
        **🤖 Predictive Analytics**
        - Trial success prediction
        - Enrollment forecasting
        - Risk assessment
        - Trend analysis
        """)

def render_dashboard_overview():
    """Render dashboard overview when database is connected"""
    
    # Database info
    if st.session_state.database_type == 'sample':
        db_info = "📊 Sample Database"
    elif st.session_state.database_type == 'local':
        db_info = "💾 Local Database"
    else:
        db_info = "🌐 Real AACT Database"
    st.sidebar.markdown(f"**Current:** {db_info}")
    
    # Welcome message
    st.markdown(f"""
    <div class="success-box">
        <h4>✅ Connected to {db_info}</h4>
        <p>You're ready to explore comprehensive clinical trial analytics. Use the navigation tabs above to access different analysis modules.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick stats
    st.subheader("📈 Quick Overview")
    
    try:
        # Get basic stats
        total_studies = st.session_state.db_manager.execute_query(
            st.session_state.config.get_query('database', 'get_study_count')
        ).iloc[0]['count']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Studies", f"{total_studies:,}")
        
        with col2:
            st.metric("Analysis Modules", "10")
        
        with col3:
            st.metric("Data Sources", db_info)
        
        with col4:
            st.metric("Last Updated", "Live")
        
    except Exception as e:
        st.error(f"Error loading overview data: {e}")
    
    # Navigation guide
    st.markdown("---")
    st.subheader("🧭 Navigation Guide")
    
    nav_col1, nav_col2,nav_col3 = st.columns(3,gap='large',vertical_alignment='top')
    
    with nav_col1:
        st.markdown("#### 📊 Analytics & Insights")
        st.markdown("---")
        st.markdown("""
        **📊 Trial Landscape**
        - Overview of trial distribution and trends
        - Phase and status analysis
        - Geographic insights
        
        **✅ Success Analytics**
        - Completion rates and success metrics
        - Results reporting analysis
        - Duration and efficiency insights
        
        **💡 Innovation Insights**
        - Intervention trends and patterns
        - Emerging therapeutic areas
        - Market opportunity analysis
        
         **🤖 Predictive Analytics**
        - Trial success prediction
        - Enrollment forecasting
        - Risk assessment models
       
        """)
    
#  **🌍 Access & Equity**
#         - Geographic access patterns
#         - Demographic inclusion analysis
#         - Diversity and equity metrics
        
#         **👨‍⚕️ Investigator Ranking**
#         - ML-powered investigator analysis
#         - Performance scoring and ranking
#         - Collaboration insights

    with nav_col2:
        st.markdown("#### 🎯 Trial Planning")
        st.markdown("---")
        st.markdown("""
        **🔍 Site Feasibility**
        - Existing Trail Metrics
        - Ranked facility recommendations with suitability scores
        - Detailed demographic breakdowns for each location
        
        **🏥 Facility Ranking**
        - Facility performance analysis
        - Efficiency and innovation metrics
        - Capacity and capability assessment
        """)

    with nav_col3:
        st.markdown("#### 📑 Reports")
        st.markdown("---")
        st.markdown("""
        **📈 Executive Summary**
        - High-level insights and trends
        - Strategic recommendations
        - Key performance indicators

        **📄 Study Reports**
        - Detailed individual study analysis
        - Comprehensive study profiles
        - Cross-study comparisons
        
        **🏢 Sponsor Profiles**
        - Sponsor performance analysis
        - Portfolio insights and trends
        - Strategic recommendations
        """)
    
    # Getting started tips
    st.markdown("---")
    st.subheader("💡 Getting Started Tips")
    
    tips_col1, tips_col2 = st.columns(2)
    
    with tips_col1:
        st.markdown("""
        **🎯 For Researchers:**
        1. Start with Trial Landscape for overview
        2. Use Investigator Ranking to find collaborators
        3. Explore Success Analytics for best practices
        4. Check Access & Equity for inclusion insights
        
        **🏢 For Industry Professionals:**
        1. Review Innovation Insights for market trends
        2. Analyze Sponsor Profiles for competitive intelligence
        3. Use Predictive Analytics for strategic planning
        4. Check Facility Ranking for site selection
        """)
    
    with tips_col2:
        st.markdown("""
        **📋 For Policymakers:**
        1. Start with Executive Summary for high-level view
        2. Review Access & Equity for policy implications
        3. Analyze Success Analytics for regulatory insights
        4. Use Trial Landscape for resource allocation
        
        **🔬 For Data Scientists:**
        1. Explore all modules for comprehensive analysis
        2. Use raw data tables for custom analysis
        3. Leverage ML models in Investigator Ranking
        4. Apply Predictive Analytics for advanced insights
        """) 

render_home_page()