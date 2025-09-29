"""
Success Analytics Page
=====================

Analyzes clinical trial success metrics including completion rates,
results reporting, and success factors.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from decimal import Decimal
from src.utils.styling import create_insight_box
from src.utils.common_service import get_success_completion_metrics,check_database_connection

def render_success_analytics_page():
    """Render the success analytics page"""
    
    st.header("✅ Success & Completion Analytics")
    
     # Check if database is connected
    if not check_database_connection():
        return
        
    
    # Load metrics
    metrics = get_success_completion_metrics()
    
    # Convert Decimal to float for metrics
    for key in metrics:
        if not metrics[key].empty:
            metrics[key] = metrics[key].apply(pd.to_numeric, errors='ignore')
    
    # Key insights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not metrics['completion_by_phase'].empty:
            best_phase = metrics['completion_by_phase'].iloc[0]
            st.metric(
                "Highest Completion Rate", 
                f"{float(best_phase['completion_rate'])}%",
                f"{best_phase['phase']}"
            )
    
    with col2:
        if not metrics['results_reporting'].empty:
            best_reporter = metrics['results_reporting'].iloc[0]
            st.metric(
                "Best Results Reporting", 
                f"{float(best_reporter['reporting_rate'])}%",
                f"{best_reporter['source_class']}"
            )
    
    with col3:
        if not metrics['duration_analysis'].empty:
            avg_duration = float(metrics['duration_analysis']['avg_duration'].mean())
            st.metric(
                "Average Trial Duration",
                f"{avg_duration:.1f} months"
            )
    
    # Visualizations
    st.subheader("📈 Success Metrics Visualizations")
    
    # Create visualization tabs
    viz_tab1, viz_tab2, viz_tab3 = st.tabs([
        "📊 Completion Analysis", 
        "📋 Results Reporting", 
        "⏱️ Duration Analysis"
    ])
    
    with viz_tab1:
        render_completion_visualizations(metrics)
    
    with viz_tab2:
        render_reporting_visualizations(metrics)
    
    with viz_tab3:
        render_duration_visualizations(metrics)
    
    # Data tables
    with st.expander("📋 Detailed Data Tables"):
        tab1, tab2, tab3 = st.tabs([
            "Completion by Phase", 
            "Results Reporting", 
            "Duration Analysis"
        ])
        
        with tab1:
            if not metrics['completion_by_phase'].empty:
                st.dataframe(metrics['completion_by_phase'], use_container_width=True)
            else:
                st.info("No completion data available")
        
        with tab2:
            if not metrics['results_reporting'].empty:
                st.dataframe(metrics['results_reporting'], use_container_width=True)
            else:
                st.info("No results reporting data available")
        
        with tab3:
            if not metrics['duration_analysis'].empty:
                st.dataframe(metrics['duration_analysis'], use_container_width=True)
            else:
                st.info("No duration data available")
    
    # Insights
    st.markdown("---")
    render_success_insights(metrics)


def render_completion_visualizations(metrics):
    """Render completion rate visualizations"""
    
    if not metrics['completion_by_phase'].empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Completion rate by phase
            fig_completion = px.bar(
                metrics['completion_by_phase'],
                x='phase',
                y='completion_rate',
                title="Completion Rate by Phase",
                color='completion_rate',
                color_continuous_scale='viridis',
                text='completion_rate'
            )
            fig_completion.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_completion.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_completion, use_container_width=True)
        
        with col2:
            # Completion vs total studies
            fig_completion_vs_total = px.scatter(
                metrics['completion_by_phase'],
                x='total_studies',
                y='completion_rate',
                size='completed_studies',
                color='phase',
                title="Completion Rate vs Total Studies by Phase",
                hover_data=['completed_studies', 'total_studies']
            )
            st.plotly_chart(fig_completion_vs_total, use_container_width=True)
        
        # Completion trend analysis
        st.subheader("Completion Trend Analysis")
        
        # Create a trend visualization (simplified)
        trend_data = metrics['completion_by_phase'].copy()
        trend_data['success_ratio'] = trend_data['completed_studies'] / trend_data['total_studies']
        
        fig_trend = px.line(
            trend_data,
            x='phase',
            y='success_ratio',
            title="Success Ratio by Phase",
            markers=True
        )
        fig_trend.update_layout(yaxis_title="Success Ratio", xaxis_title="Phase")
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Completion insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Completion Insights:**")
            best_phase = metrics['completion_by_phase'].iloc[0]
            worst_phase = metrics['completion_by_phase'].iloc[-1]
            
            st.markdown(f"""
            - **Best Performing Phase**: {best_phase['phase']} ({float(best_phase['completion_rate']):.1f}%)
            - **Challenging Phase**: {worst_phase['phase']} ({float(worst_phase['completion_rate']):.1f}%)
            - **Average Completion**: {float(metrics['completion_by_phase']['completion_rate'].mean()):.1f}%
            """)
        
        with col2:
            st.markdown("**🎯 Success Factors:**")
            st.markdown("""
            - **Phase 1**: Higher completion due to smaller scale
            - **Phase 3**: Lower completion due to complexity
            - **Sample Size**: Larger studies show varied completion rates
            - **Duration**: Longer trials face higher attrition
            """)
    else:
        st.info("No completion data available")

def render_reporting_visualizations(metrics):
    """Render results reporting visualizations"""
    
    if not metrics['results_reporting'].empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Reporting rate by sponsor type
            fig_reporting = px.bar(
                metrics['results_reporting'],
                x='source_class',
                y='reporting_rate',
                title="Results Reporting Rate by Sponsor Type",
                color='reporting_rate',
                color_continuous_scale='plasma',
                text='reporting_rate'
            )
            fig_reporting.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_reporting.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_reporting, use_container_width=True)
        
        with col2:
            # Reporting vs total studies
            fig_reporting_vs_total = px.scatter(
                metrics['results_reporting'],
                x='total_studies',
                y='reporting_rate',
                size='reported_results',
                color='source_class',
                title="Reporting Rate vs Total Studies by Sponsor",
                hover_data=['reported_results', 'total_studies']
            )
            st.plotly_chart(fig_reporting_vs_total, use_container_width=True)
        
        # Time to report analysis
        st.subheader("Time to Report Results")
        
        if 'avg_months_to_report' in metrics['results_reporting'].columns:
            fig_time = px.bar(
                metrics['results_reporting'],
                x='source_class',
                y='avg_months_to_report',
                title="Average Months to Report Results by Sponsor Type",
                color='avg_months_to_report',
                color_continuous_scale='viridis'
            )
            fig_time.update_layout(xaxis_tickangle=-45, yaxis_title="Average Months")
            st.plotly_chart(fig_time, use_container_width=True)
        
        # Reporting compliance insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📋 Reporting Insights:**")
            best_reporter = metrics['results_reporting'].iloc[0]
            worst_reporter = metrics['results_reporting'].iloc[-1]
            
            st.markdown(f"""
            - **Best Reporter**: {best_reporter['source_class']} ({float(best_reporter['reporting_rate']):.1f}%)
            - **Needs Improvement**: {worst_reporter['source_class']} ({float(worst_reporter['reporting_rate']):.1f}%)
            - **Average Reporting**: {float(metrics['results_reporting']['reporting_rate'].mean()):.1f}%
            """)
        
        with col2:
            st.markdown("**⏱️ Timing Insights:**")
            if 'avg_months_to_report' in metrics['results_reporting'].columns:
                fastest = metrics['results_reporting'].loc[metrics['results_reporting']['avg_months_to_report'].idxmin()]
                slowest = metrics['results_reporting'].loc[metrics['results_reporting']['avg_months_to_report'].idxmax()]
                
                st.markdown(f"""
                - **Fastest Reporter**: {fastest['source_class']} ({float(fastest['avg_months_to_report']):.1f} months)
                - **Slowest Reporter**: {slowest['source_class']} ({float(slowest['avg_months_to_report']):.1f} months)
                - **Average Time**: {float(metrics['results_reporting']['avg_months_to_report'].mean()):.1f} months
                """)
    else:
        st.info("No results reporting data available")

def render_duration_visualizations(metrics):
    """Render duration analysis visualizations"""
    
    if not metrics['duration_analysis'].empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Average duration by phase
            fig_duration = px.bar(
                metrics['duration_analysis'],
                x='phase',
                y='avg_duration',
                title="Average Trial Duration by Phase",
                color='avg_duration',
                color_continuous_scale='viridis',
                text='avg_duration'
            )
            fig_duration.update_traces(texttemplate='%{text:.1f} months', textposition='outside')
            fig_duration.update_layout(xaxis_tickangle=-45, yaxis_title="Average Duration (months)")
            st.plotly_chart(fig_duration, use_container_width=True)
        
        with col2:
            # Duration range by phase
            duration_data = metrics['duration_analysis'].melt(
                id_vars=['phase'],
                value_vars=['min_duration', 'avg_duration', 'max_duration'],
                var_name='metric',
                value_name='duration'
            )
            fig_duration_range = px.bar(
                duration_data,
                x='phase',
                y='duration',
                color='metric',
                title="Duration Range by Phase",
                barmode='group'
            )
            fig_duration_range.update_layout(xaxis_tickangle=-45, yaxis_title="Duration (months)")
            st.plotly_chart(fig_duration_range, use_container_width=True)
        # Duration vs completion correlation
        st.subheader("Duration vs Completion Correlation")
        
        # Merge duration and completion data
        if not metrics['completion_by_phase'].empty:
            merged_data = pd.merge(
                metrics['duration_analysis'],
                metrics['completion_by_phase'],
                on='phase',
                how='inner'
            )
            
            fig_correlation = px.scatter(
                merged_data,
                x='avg_duration',
                y='completion_rate',
                size='studies_count',
                color='phase',
                title="Duration vs Completion Rate Correlation",
                hover_data=['studies_count', 'total_studies']
            )
            fig_correlation.update_layout(
                xaxis_title="Average Duration (months)",
                yaxis_title="Completion Rate (%)"
            )
            st.plotly_chart(fig_correlation, use_container_width=True)
        
        # Duration insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**⏱️ Duration Insights:**")
            shortest_phase = metrics['duration_analysis'].loc[metrics['duration_analysis']['avg_duration'].idxmin()]
            longest_phase = metrics['duration_analysis'].loc[metrics['duration_analysis']['avg_duration'].idxmax()]
            
            st.markdown(f"""
            - **Shortest Trials**: {shortest_phase['phase']} ({float(shortest_phase['avg_duration']):.1f} months)
            - **Longest Trials**: {longest_phase['phase']} ({float(longest_phase['avg_duration']):.1f} months)
            - **Average Duration**: {float(metrics['duration_analysis']['avg_duration'].mean()):.1f} months
            """)
        
        with col2:
            st.markdown("**📊 Duration Patterns:**")
            st.markdown("""
            - **Phase 1**: Typically shortest due to safety focus
            - **Phase 2**: Moderate duration for efficacy
            - **Phase 3**: Longest due to large sample sizes
            - **Variability**: High standard deviation indicates diverse trial designs
            """)
    else:
        st.info("No duration data available")

def render_success_insights(metrics):
    """Render success insights and analysis"""
    
    st.subheader("🔍 Success Analytics Insights")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        # Completion insights
        if not metrics['completion_by_phase'].empty:
            best_phase = metrics['completion_by_phase'].iloc[0]
            avg_completion = float(metrics['completion_by_phase']['completion_rate'].mean())
            
            st.markdown(f"""
            **✅ Completion Insights:**
            - **Best Phase**: {best_phase['phase']} with {float(best_phase['completion_rate']):.1f}% completion
            - **Average Completion**: {avg_completion:.1f}% across all phases
            - **Phase Variation**: {len(metrics['completion_by_phase'])} phases show different success rates
            - **Success Factors**: Early phases typically show higher completion rates
            """)
        
        # Duration insights
        if not metrics['duration_analysis'].empty:
            avg_duration = float(metrics['duration_analysis']['avg_duration'].mean())
            duration_variation = float(metrics['duration_analysis']['avg_duration'].std())
            
            st.markdown(f"""
            **⏱️ Duration Insights:**
            - **Average Duration**: {avg_duration:.1f} months across all phases
            - **Duration Variation**: {duration_variation:.1f} months standard deviation
            - **Phase Patterns**: Clear progression from short to long trials
            - **Efficiency**: Duration correlates with trial complexity and sample size
            """)
    
    with insights_col2:
        # Reporting insights
        if not metrics['results_reporting'].empty:
            best_reporter = metrics['results_reporting'].iloc[0]
            avg_reporting = float(metrics['results_reporting']['reporting_rate'].mean())
            
            st.markdown(f"""
            **📋 Results Reporting Insights:**
            - **Best Reporter**: {best_reporter['source_class']} with {float(best_reporter['reporting_rate']):.1f}% compliance
            - **Average Reporting**: {avg_reporting:.1f}% across all sponsor types
            - **Compliance Gap**: Significant variation in reporting practices
            - **Regulatory Impact**: Industry sponsors often show better compliance
            """)
        
        # Success factors
        st.markdown("""
        **🎯 Key Success Factors:**
        - **Phase Selection**: Early phases show higher completion rates
        - **Sponsor Type**: Industry sponsors often perform better
        - **Study Design**: Simpler designs correlate with success
        - **Geographic Distribution**: Local trials show better completion
        - **Sample Size**: Optimal size balances power and feasibility
        """)
    
    # Recommendations
    st.markdown("---")
    st.subheader("💡 Strategic Recommendations")
    
    recommendations = [
        "**Optimize Phase Selection**: Focus resources on phases with proven success rates",
        "**Improve Reporting Compliance**: Implement mandatory reporting timelines and incentives",
        "**Streamline Trial Design**: Simplify protocols to reduce duration and improve completion",
        "**Enhance Sponsor Collaboration**: Foster partnerships between different sponsor types",
        "**Implement Risk Mitigation**: Develop strategies for high-risk phases and sponsor types",
        "**Monitor Success Metrics**: Establish regular tracking of completion and reporting rates",
        "**Invest in Technology**: Use digital solutions to improve trial efficiency and compliance"
    ]
    
    for rec in recommendations:
        st.markdown(f"• {rec}")
    
    # Success prediction model (placeholder)
    st.markdown("---")
    st.subheader("🤖 Success Prediction Model")
    
    st.markdown("""
    **Advanced Analytics Features:**
    - **ML-powered success prediction** based on historical data
    - **Risk assessment models** for trial planning
    - **Optimization algorithms** for trial design
    - **Real-time monitoring** of success indicators
    
    *These features are available in the Predictive Analytics module.*
    """) 

render_success_analytics_page()