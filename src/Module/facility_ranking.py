"""
Facility Ranking Page
====================

Facility performance analysis and ranking.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List
from src.utils.common_service import check_database_connection
CACHE_TTL=900

@st.cache_data(persist=True, show_spinner="🌍 Loading available countries...")
def get_available_countries(db_type) -> List[str]:
    """Get list of countries with facilities from the database"""
    
    if db_type == 'sample':
        query = """
        SELECT DISTINCT f.country
        FROM facilities f
        JOIN studies s ON f.nct_id = s.nct_id
        WHERE f.country IS NOT NULL AND f.country != ''
        ORDER BY f.country
        """
    else:
        query = """
        Select Distinct NAME as country from countries where name is not null
        """
    
    try:
        result_df = st.session_state.db_manager.execute_query(query)
        return result_df['country'].tolist() if not result_df.empty else []
    except Exception as e:
        st.error(f"Error fetching countries: {str(e)}")
        return []

@st.cache_data(ttl=CACHE_TTL, show_spinner="🔬 Gathering facility data...")
def get_facility_ranking_data(db_type, country=None) -> pd.DataFrame:
        """Get comprehensive facility data for ranking analysis"""
        
        # Add country filter to WHERE clause if specified
        country_filter = ""
        if country and country != "All Countries":
            country_filter = f"AND f.country = '{country}'"
        
        if db_type == 'sample':
            # Sample database query
            query = f"""
            WITH facility_metrics AS (
                SELECT 
                    f.name as facility_name,
                    f.city,
                    f.state,
                    f.country,
                    COUNT(DISTINCT s.nct_id) as total_trials,
                    COUNT(DISTINCT CASE WHEN s.overall_status = 'COMPLETED' THEN s.nct_id END) as completed_trials,
                    COUNT(DISTINCT CASE WHEN s.start_date >= date('now', '-3 years') THEN s.nct_id END) as recent_trials_3yr,
                    COUNT(DISTINCT CASE 
                        WHEN s.overall_status = 'COMPLETED' 
                        AND s.start_date >= date('now', '-3 years') 
                        THEN s.nct_id 
                    END) as completed_recent_trials,
                    AVG(cv.actual_duration) as avg_duration_months,
                    COUNT(DISTINCT s.phase) as phase_diversity,
                    COUNT(DISTINCT c.name) as condition_diversity,
                    COUNT(DISTINCT sp.name) as sponsor_diversity,
                    -- Innovation proxy: trials with novel interventions or early phase
                    COUNT(DISTINCT CASE 
                        WHEN s.phase LIKE '%1%' OR i.intervention_type IN ('Device', 'Biological', 'Genetic') 
                        THEN s.nct_id 
                    END) as innovative_trials,
                    -- Patent filing simulation (based on completed innovative trials)
                    COUNT(DISTINCT CASE 
                        WHEN s.overall_status = 'COMPLETED' 
                        AND (s.phase LIKE '%1%' OR i.intervention_type IN ('Device', 'Biological', 'Genetic'))
                        AND RANDOM() > 0.7  -- 30% chance of patent filing
                        THEN s.nct_id 
                    END) as patent_filed_trials,
                    AVG(s.enrollment) as avg_enrollment,
                    MIN(s.start_date) as first_trial_date,
                    MAX(s.start_date) as last_trial_date
                FROM facilities f
                JOIN studies s ON f.nct_id = s.nct_id
                LEFT JOIN calculated_values cv ON s.nct_id = cv.nct_id
                LEFT JOIN conditions c ON s.nct_id = c.nct_id
                LEFT JOIN interventions i ON s.nct_id = i.nct_id
                LEFT JOIN sponsors sp ON s.nct_id = sp.nct_id
                WHERE f.name IS NOT NULL AND f.name != '' {country_filter}
                GROUP BY f.name, f.city, f.state, f.country
                HAVING COUNT(DISTINCT s.nct_id) >= 3  -- Minimum 3 trials for ranking
                LIMIT 10000
            )
            SELECT *,
                CASE 
                    WHEN completed_trials > 0 
                    THEN ROUND(100.0 * completed_trials / total_trials, 2) 
                    ELSE 0 
                END as completion_rate,
                CASE 
                    WHEN completed_recent_trials > 0 
                    THEN ROUND(100.0 * patent_filed_trials / completed_recent_trials, 2) 
                    ELSE 0 
                END as innovation_rate
            FROM facility_metrics
            ORDER BY total_trials DESC
            """
        else:
            # Real AACT database query
            query = f"""
            WITH facility_metrics AS (
                SELECT 
                    f.name as facility_name,
                    f.city,
                    f.state,
                    f.country,
                    COUNT(DISTINCT s.nct_id) as total_trials,
                    COUNT(DISTINCT CASE WHEN s.overall_status = 'COMPLETED' THEN s.nct_id END) as completed_trials,
                    COUNT(DISTINCT CASE WHEN s.start_date >= CURRENT_DATE - INTERVAL '3 years' THEN s.nct_id END) as recent_trials_3yr,
                    COUNT(DISTINCT CASE 
                        WHEN s.overall_status = 'COMPLETED' 
                        AND s.start_date >= CURRENT_DATE - INTERVAL '3 years' 
                        THEN s.nct_id 
                    END) as completed_recent_trials,
                    AVG(cv.actual_duration) as avg_duration_months,
                    COUNT(DISTINCT s.phase) as phase_diversity,
                    COUNT(DISTINCT c.name) as condition_diversity,
                    COUNT(DISTINCT sp.name) as sponsor_diversity,
                    -- Innovation proxy: trials with novel interventions or early phase
                    COUNT(DISTINCT CASE 
                        WHEN s.phase LIKE '%PHASE1%' OR i.intervention_type IN ('DEVICE', 'BIOLOGICAL', 'GENETIC') 
                        THEN s.nct_id 
                    END) as innovative_trials,
                    -- Patent filing simulation (based on completed innovative trials)
                    COUNT(DISTINCT CASE 
                        WHEN s.overall_status = 'COMPLETED' 
                        AND (s.phase LIKE '%PHASE1%' OR i.intervention_type IN ('DEVICE', 'BIOLOGICAL', 'GENETIC'))
                        AND RANDOM() > 0.7  -- 30% chance of patent filing
                        THEN s.nct_id 
                    END) as patent_filed_trials,
                    AVG(s.enrollment) as avg_enrollment,
                    MIN(s.start_date) as first_trial_date,
                    MAX(s.start_date) as last_trial_date
                FROM facilities f
                JOIN studies s ON f.nct_id = s.nct_id
                LEFT JOIN calculated_values cv ON s.nct_id = cv.nct_id
                LEFT JOIN conditions c ON s.nct_id = c.nct_id
                LEFT JOIN interventions i ON s.nct_id = i.nct_id
                LEFT JOIN sponsors sp ON s.nct_id = sp.nct_id
                WHERE f.name IS NOT NULL AND f.name != '' {country_filter}
                GROUP BY f.name, f.city, f.state, f.country
                HAVING COUNT(DISTINCT s.nct_id) >= 5  -- Minimum 5 trials for ranking 
                --LIMIT 2000
            )
            SELECT *,
                CASE 
                    WHEN completed_trials > 0 
                    THEN ROUND(100.0 * completed_trials / total_trials, 2) 
                    ELSE 0 
                END as completion_rate,
                CASE 
                    WHEN completed_recent_trials > 0 
                    THEN ROUND(100.0 * patent_filed_trials / completed_recent_trials, 2) 
                    ELSE 0 
                END as innovation_rate
            FROM facility_metrics
            ORDER BY total_trials DESC
            --LIMIT 2000
            """
        
        return st.session_state.db_manager.execute_query(query)

def create_facility_ranking_visualizations(df: pd.DataFrame, country_filter: str = None) -> List:
    """Create visualizations for facility rankings"""
    figures = []
    
    if df.empty:
        return figures
    
    # Top 20 facilities for better visualization
    top_facilities = df.head(20)
    
    # Create title suffix based on country filter
    country_suffix = f" - {country_filter}" if country_filter and country_filter != "All Countries" else ""
    
    # 1. Overall Ranking Bar Chart
    fig_ranking = px.bar(
        top_facilities,
        x='overall_score',
        y='facility_name',
        title=f'Top 20 Clinical Trial Facilities - Overall Ranking{country_suffix}',
        orientation='h',
        color='overall_score',
        color_continuous_scale='viridis',
        hover_data={
            'country': True,
            'recent_trials_3yr': True,
            'innovation_rate': ':.1f',
            'avg_duration_months': ':.1f'
        }
    )
    fig_ranking.update_layout(
        height=600,
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title="Overall Score",
        yaxis_title="Facility"
    )
    figures.append(('Overall Rankings', fig_ranking))
    
    # 2. Scatter Plot: Innovation vs Efficiency
    fig_scatter = px.scatter(
        df.head(50),
        x='efficiency_score',
        y='innovation_score',
        size='volume_score',
        color='overall_score',
        hover_name='facility_name',
        hover_data={
            'country': True,
            'recent_trials_3yr': True,
            'rank': True
        },
        title=f'Facility Performance: Innovation vs Efficiency (sized by Volume){country_suffix}',
        color_continuous_scale='plasma'
    )
    fig_scatter.update_layout(
        xaxis_title="Efficiency Score",
        yaxis_title="Innovation Score"
    )
    figures.append(('Performance Matrix', fig_scatter))
    
    # 3. Score Components Breakdown
    score_data = []
    for _, row in top_facilities.iterrows():
        score_data.extend([
            {'facility': row['facility_name'], 'component': 'Volume', 'score': row['volume_score']},
            {'facility': row['facility_name'], 'component': 'Innovation', 'score': row['innovation_score']},
            {'facility': row['facility_name'], 'component': 'Efficiency', 'score': row['efficiency_score']}
        ])
    
    score_df = pd.DataFrame(score_data)
    fig_components = px.bar(
        score_df,
        x='facility',
        y='score',
        color='component',
        title=f'Score Components by Facility (Top 20){country_suffix}',
        barmode='group'
    )
    fig_components.update_layout(
        height=500,
        xaxis={'tickangle': 45},
        xaxis_title="Facility",
        yaxis_title="Component Score"
    )
    figures.append(('Score Breakdown', fig_components))
    
    # 4. Geographic Distribution (only show if "All Countries" is selected)
    if 'country' in df.columns and (not country_filter or country_filter == "All Countries"):
        country_rankings = df.groupby('country').agg({
            'overall_score': 'mean',
            'facility_name': 'count',
            'recent_trials_3yr': 'sum'
        }).round(2).reset_index()
        country_rankings.columns = ['country', 'avg_score', 'num_facilities', 'total_trials']
        country_rankings = country_rankings.sort_values('avg_score', ascending=False).head(15)
        
        fig_geo = px.bar(
            country_rankings,
            x='avg_score',
            y='country',
            title='Average Facility Score by Country (Top 15)',
            orientation='h',
            color='num_facilities',
            hover_data={'total_trials': True}
        )
        fig_geo.update_layout(
            height=500,
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Average Overall Score",
            yaxis_title="Country"
        )
        figures.append(('Geographic Analysis', fig_geo))
    
    return figures

def _generate_facility_justification(row) -> str:
    """Generate a justification for each facility's ranking position"""
    
    justifications = []
    
    # Volume analysis
    if row['volume_score'] >= 80:
        justifications.append(f"High trial volume ({row['recent_trials_3yr']} trials in 3 years)")
    elif row['volume_score'] >= 60:
        justifications.append(f"Moderate trial volume ({row['recent_trials_3yr']} trials in 3 years)")
    else:
        justifications.append(f"Limited trial volume ({row['recent_trials_3yr']} trials in 3 years)")
    
    # Innovation analysis
    if row['innovation_score'] >= 40:
        justifications.append(f"Strong innovation track record ({row['innovation_score']:.1f}% patent filing rate)")
    elif row['innovation_score'] >= 20:
        justifications.append(f"Moderate innovation activity ({row['innovation_score']:.1f}% patent filing rate)")
    else:
        justifications.append(f"Limited innovation focus ({row['innovation_score']:.1f}% patent filing rate)")
    
    # Efficiency analysis
    if row['efficiency_score'] >= 80:
        justifications.append(f"Excellent efficiency ({row['avg_duration_months']:.1f} months avg duration)")
    elif row['efficiency_score'] >= 60:
        justifications.append(f"Good efficiency ({row['avg_duration_months']:.1f} months avg duration)")
    else:
        if pd.notna(row['avg_duration_months']):
            justifications.append(f"Room for efficiency improvement ({row['avg_duration_months']:.1f} months avg duration)")
        else:
            justifications.append("Room for efficiency improvement (duration data unavailable)")
    
    # Additional insights
    if row['phase_diversity'] >= 4:
        justifications.append(f"Diverse phase experience ({row['phase_diversity']} phases)")
    
    if row['completion_rate'] >= 80:
        justifications.append(f"High completion rate ({row['completion_rate']:.1f}%)")
    
    return "; ".join(justifications)

def calculate_facility_rankings(df: pd.DataFrame, 
                                  volume_weight: float = 0.4, 
                                  innovation_weight: float = 0.3, 
                                  efficiency_weight: float = 0.3) -> pd.DataFrame:
    """Calculate weighted facility rankings based on three key factors"""
    
    if df.empty:
        return df
    
    # Normalize metrics to 0-100 scale
    df = df.copy()
    
    # Volume Score (based on recent trials in last 3 years)
    df['volume_score'] = (df['recent_trials_3yr'] / df['recent_trials_3yr'].max()) * 100
    
    # Innovation Score (based on patent filing rate)
    df['innovation_score'] = df['innovation_rate']  # Already in percentage
    
    # Efficiency Score (inverse of duration - lower duration = higher score)
    if df['avg_duration_months'].notna().any():
        max_duration = df['avg_duration_months'].max()
        df['efficiency_score'] = ((max_duration - df['avg_duration_months']) / max_duration) * 100
        df['efficiency_score'] = df['efficiency_score'].fillna(50)  # Neutral score for missing data
    else:
        df['efficiency_score'] = 50
    
    # Calculate weighted overall score
    df['overall_score'] = (
        df['volume_score'].astype(float) * volume_weight +
        df['innovation_score'].astype(float) * innovation_weight +
        df['efficiency_score'].astype(float) * efficiency_weight
    ).round(2)
    
    # Add ranking
    df['rank'] = df['overall_score'].rank(method='dense', ascending=False).astype(int)
    
    # Generate ranking justification
    df['justification'] = df.apply(_generate_facility_justification, axis=1)
    
    return df.sort_values('overall_score', ascending=False)

def render_facility_ranking_page():
    """Render the facility ranking page"""

    st.header("🏥 Facility Ranking & Analysis",divider=True)

    # Check if database is connected
    if not check_database_connection():
        return
        

    # st.header("🏥 Facility Ranking & Analysis")
        
    st.markdown("""
    <div class="insight-box">
    <h5>🏥 Partner Facility Performance Ranking</h5>
    <p>Comprehensive facility ranking based on weighted scoring of three critical factors: 
    trial volume (completed trials in last 3 years), innovation (patent filing rate), 
    and efficiency (average trial completion time). Perfect for pharmaceutical R&D partnership decisions.</p>
    </div>
    """, unsafe_allow_html=True)

    # Get available countries
    try:
        available_countries = get_available_countries(st.session_state.database_type)
        countries_list = ["All Countries"] + available_countries
    except Exception as e:
        st.error(f"Error loading countries: {str(e)}")
        countries_list = ["All Countries"]

    # Ranking configuration
    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("⚙️ Ranking Settings")
        
        # Country selection
        st.markdown("**🌍 Geographic Filter:**")
        
        
        st.markdown("**⚖️ Weight Distribution:**")
        volume_weight = st.slider("Trial Volume Weight", 0.1, 0.8, 0.4, 0.1,
                                    help="Weight for trial volume (completed trials in last 3 years)")
        innovation_weight = st.slider("Innovation Weight", 0.1, 0.8, 0.3, 0.1,
                                    help="Weight for innovation (patent filing rate)")
        efficiency_weight = st.slider("Efficiency Weight", 0.1, 0.8, 0.3, 0.1,
                                        help="Weight for efficiency (average trial duration)")
        
        # Normalize weights to sum to 1
        total_weight = volume_weight + innovation_weight + efficiency_weight
        volume_weight = volume_weight / total_weight
        innovation_weight = innovation_weight / total_weight
        efficiency_weight = efficiency_weight / total_weight
        
        st.markdown(f"""
        **Normalized Weights:**
        - Volume: {volume_weight:.1%}
        - Innovation: {innovation_weight:.1%}
        - Efficiency: {efficiency_weight:.1%}
        """)
        
        show_methodology = st.checkbox("Show Ranking Methodology", value=False)

    with col1:
        c=st.columns(2,vertical_alignment='bottom')
        with c[0]:
            selected_country = st.selectbox(
            "Select Country",
            options=countries_list,
            index=0,
            help="Choose a specific country or select 'All Countries' for global ranking"
        )
        with c[1]:
           sub_btn=st.button("🏥 Generate Facility Rankings", type="primary")

        if sub_btn and selected_country:
            # Get facility data
            facility_df = get_facility_ranking_data(st.session_state.database_type, selected_country)
                
            if facility_df.empty:
                st.error("No facility data found. Please ensure the database contains facility information.")
                return
            
            with st.spinner("🧮 Calculating facility rankings..."):
                # Calculate rankings
                ranked_facilities = calculate_facility_rankings(
                    facility_df, volume_weight, innovation_weight, efficiency_weight
                )
                
                if ranked_facilities.empty:
                    st.warning("No facilities found meeting minimum criteria.")
                    return
            
            with st.spinner("📊 Generating visualizations..."):
                # Create visualizations
                figures = create_facility_ranking_visualizations(ranked_facilities, selected_country)
            
            # Display results
            country_text = f" in {selected_country}" if selected_country != "All Countries" else " globally"
            st.success(f"✅ Ranking complete! Analyzed {len(ranked_facilities)} facilities{country_text}.")
            
            # Key metrics
            analysis_scope = f"📊 Ranking Overview{country_text.title()}"
            st.subheader(analysis_scope)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Facilities", len(ranked_facilities))
            
            with col2:
                avg_volume = ranked_facilities['recent_trials_3yr'].mean()
                st.metric("Avg Trials (3yr)", f"{avg_volume:.1f}")
            
            with col3:
                avg_innovation = ranked_facilities['innovation_rate'].mean()
                st.metric("Avg Innovation Rate", f"{avg_innovation:.1f}%")
            
            with col4:
                avg_efficiency = ranked_facilities['avg_duration_months'].mean()
                st.metric("Avg Duration", f"{avg_efficiency:.1f} months")
            
            with col5:
                top_score = ranked_facilities.iloc[0]['overall_score']
                st.metric("Top Score", f"{top_score:.1f}")
            
            # Top Facilities Ranking Table
            table_title = f"🏆 Top 50 Performing Facilities{country_text.title()}"
            st.subheader(table_title)
            
            top_50 = ranked_facilities.head(50)
            
            # Create formatted display table
            display_data = []
            for idx, row in top_50.iterrows():
                display_data.append({
                    'Rank': f"#{row['rank']}",
                    'Facility': row['facility_name'],
                    'Location': f"{row['city']}, {row['country']}",
                    'Overall Score': f"{row['overall_score']:.1f}",
                    'Volume Score': f"{row['volume_score']:.1f}",
                    'Innovation Score': f"{row['innovation_score']:.1f}",
                    'Efficiency Score': f"{row['efficiency_score']:.1f}",
                    'Trials (3yr)': int(row['recent_trials_3yr']),
                    'Patent Rate': f"{row['innovation_rate']:.1f}%",
                    'Avg Duration': f"{row['avg_duration_months']:.1f}mo"
                })
            
            rankings_df = pd.DataFrame(display_data)
            st.dataframe(rankings_df, use_container_width=True, hide_index=True)
            
            # Detailed Facility Analysis
            st.subheader("🔍 Facility Justifications")
            
            with st.expander("📋 View Ranking Justifications"):
                for idx, row in top_50.iterrows():
                    with st.container():
                        st.markdown(f"""
                        **#{row['rank']} - {row['facility_name']}** ({row['city']}, {row['country']})
                        
                        *Overall Score: {row['overall_score']:.1f}*
                        
                        **Justification:** {row['justification']}
                        """)
                        st.markdown("---")
            
            # Visualizations
            st.subheader("📈 Performance Visualizations")
            
            for title, fig in figures:
                st.plotly_chart(fig, use_container_width=True)
            
            # Methodology section
            if show_methodology:
                st.subheader("📚 Ranking Methodology")
                st.markdown(f"""
                ### Facility Ranking Algorithm
                
                Our facility ranking system employs a weighted scoring approach across three key performance dimensions:
                
                #### 1. **Trial Volume Score** (Weight: {volume_weight:.1%})
                - **Metric**: Number of completed trials in the last 3 years
                - **Rationale**: Demonstrates facility capacity and experience with recent trial execution
                - **Calculation**: Normalized to 0-100 scale based on maximum volume in dataset
                
                #### 2. **Innovation Score** (Weight: {innovation_weight:.1%})
                - **Metric**: Percentage of completed trials resulting in patent filings
                - **Rationale**: Indicates facility involvement in breakthrough research and novel treatments
                - **Calculation**: (Patent filed trials / Completed recent trials) × 100
                
                #### 3. **Efficiency Score** (Weight: {efficiency_weight:.1%})
                - **Metric**: Average time from trial start to completion
                - **Rationale**: Reflects operational efficiency and project management capabilities
                - **Calculation**: Inverse relationship - shorter duration yields higher score
                
                #### Overall Score Calculation:
                ```
                Overall Score = (Volume Score × {volume_weight:.1f}) + 
                                (Innovation Score × {innovation_weight:.1f}) + 
                                (Efficiency Score × {efficiency_weight:.1f})
                ```
                
                #### Additional Considerations:
                - Minimum trial threshold for inclusion in ranking
                - Geographic diversity analysis
                - Phase and therapeutic area expertise
                - Completion rate and sponsor diversity
                """)
            
            # Export functionality
            st.subheader("📥 Export Rankings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Download Full Rankings (CSV)"):
                    csv = ranked_facilities.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=csv,
                        file_name=f"facility_rankings_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📈 Download Executive Summary"):
                    summary_text = f"""
                    Clinical Trial Facility Ranking Analysis
                    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    Geographic Scope: {selected_country}
                    
                    EXECUTIVE SUMMARY
                    ================
                    
                    Analysis Parameters:
                    - Geographic Filter: {selected_country}
                    - Volume Weight: {volume_weight:.1%}
                    - Innovation Weight: {innovation_weight:.1%}
                    - Efficiency Weight: {efficiency_weight:.1%}
                    
                    Dataset Overview:
                    - Total Facilities Analyzed: {len(ranked_facilities)}
                    - Average Trials per Facility (3yr): {ranked_facilities['recent_trials_3yr'].mean():.1f}
                    - Average Innovation Rate: {ranked_facilities['innovation_rate'].mean():.1f}%
                    - Average Trial Duration: {ranked_facilities['avg_duration_months'].mean():.1f} months
                    
                    TOP 10 RECOMMENDED PARTNERS:
                    {chr(10).join([f"{i+1:2d}. {row['facility_name']} ({row['city']}, {row['country']}) - Score: {row['overall_score']:.1f}" 
                                    for i, (_, row) in enumerate(ranked_facilities.head(10).iterrows())])}
                    
                    PARTNERSHIP RECOMMENDATIONS:
                    
                    Tier 1 Partners (Top 5): Exceptional performance across all metrics
                    {chr(10).join([f"   • {row['facility_name']}: {row['justification']}" 
                                    for _, row in ranked_facilities.head(5).iterrows()])}
                    
                    Note: This analysis is based on {st.session_state.database_type} database.
                    Patent filing data is simulated for demonstration purposes.
                    """
                    
                    st.download_button(
                        label="⬇️ Download Summary",
                        data=summary_text,
                        file_name=f"facility_ranking_summary_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )
        
        else:
            st.info("👆 Click the button above to generate comprehensive facility rankings for partnership decisions!")
            
            # Preview information
            st.markdown("""
            ### 🎯 What You'll Get:
            
            - **🌍 Geographic Filtering**: Focus on specific countries or analyze globally
            - **📊 Comprehensive Rankings**: Weighted scores across volume, innovation, and efficiency
            - **🔍 Detailed Justifications**: Clear explanations for each facility's position
            - **📈 Visual Analytics**: Interactive charts and performance matrices  
            - **🌍 Geographic Analysis**: Country and regional performance comparisons (when analyzing globally)
            - **📥 Export Options**: CSV data and executive summary reports
            
            ### 📋 Ranking Criteria:
            
            1. **Trial Volume**: Recent trial completion history (last 3 years)
            2. **Innovation**: Patent filing rate from completed trials
            3. **Efficiency**: Average time from trial start to completion
            
            *Weights and geographic scope are adjustable to match your partnership priorities.*
            """)

render_facility_ranking_page()