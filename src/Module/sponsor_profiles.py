"""
Sponsor Profiles Page
====================

Sponsor analysis and profile management.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from utils import pharma_gen 
from utils.common_service import get_sponsors_list,get_sponsors_stock_data,check_database_connection
from typing import Dict, List

CACHE_TTL=60*60*24

def sponsor_selection_widget():
    with st.spinner("Loading sponsors..."):
        sponsors_df = get_sponsors_list()
    if not sponsors_df.empty:
        
        selected_sponsor_display = st.selectbox(
            "Select Sponsor:",
            options=sponsors_df['sponsor_display'],index=None,key="sponsor_selection",
            help="Choose a sponsor to view their studies"
        )
        if not selected_sponsor_display:    
            return None, None,pd.DataFrame()
        else:
            # Extract actual sponsor name
            selected_sponsor_name = selected_sponsor_display.split(' [')[0]
            sp_stock_df= get_sponsors_stock_data(selected_sponsor_name)
            # if not sp_stock_df.empty:
            #     st.dataframe(sp_stock_df,use_container_width=True,hide_index=True)
            # Display sponsor info
            sponsor_info = sponsors_df[sponsors_df['sponsor_name'] == selected_sponsor_name].iloc[0]
            
            return selected_sponsor_name, sponsor_info,sp_stock_df
    else:
        st.error("No sponsors found in the database.")
        return None, None,pd.DataFrame()

@st.fragment
def render_sponsor_quick_overview(selected_sponsor_name, sponsor_info, sp_stock_df):
    st.markdown(f"""
    <div class="insight-box">
    <h4><img src="{sp_stock_df.iloc[0]['image'] if not sp_stock_df.empty else None}" alt=📊 style="max-height:40px; margin-bottom:10px;"> Quick Overview </h4>
    <p>{sp_stock_df.iloc[0]['description'] if not sp_stock_df.empty else ''}</p>
    </div>
    """, unsafe_allow_html=True)
    sp_c = st.columns(4, vertical_alignment="bottom")
    with sp_c[0]:
        st.markdown(f"""
                    **Selected Sponsor:** {sponsor_info['sponsor_name']}  
                    **Agency Class:** {sponsor_info['agency_class']}  
                    **Total Studies:** {sponsor_info['total_studies']}  
                    **Completed Studies:** {sponsor_info['completed_studies']}
                    """)
    with sp_c[1]:
        if selected_sponsor_name:
            st.markdown(f"""
            **Agency Class:** {sponsor_info['agency_class']}  
            **Total Studies:** {sponsor_info['total_studies']}  
            **Completed:** {sponsor_info['completed_studies']}  
            **Completion Rate:** {(sponsor_info['completed_studies']/sponsor_info['total_studies']*100):.1f}%
            """)
    if not sp_stock_df.empty:
        with sp_c[2]:
            st.markdown(f"""
            **Company:** {sp_stock_df.iloc[0]['companyName']}  
            **Industry:** {sp_stock_df.iloc[0]['industry']}  
            **Sector:** {sp_stock_df.iloc[0]['sector']}  
            **Country:** {sp_stock_df.iloc[0]['country']}  
            """)
        with sp_c[3]:
            st.markdown(f"""
            **Website:** {sp_stock_df.iloc[0]['website']}   
            **CEO:** {sp_stock_df.iloc[0]['ceo']}  
            **Employees:** {int(sp_stock_df.iloc[0]['fullTimeEmployees']):,}   
            **Headquarters:** {sp_stock_df.iloc[0]['address']}, {sp_stock_df.iloc[0]['city']}, {sp_stock_df.iloc[0]['state']}, {sp_stock_df.iloc[0]['zip']}  
            """)
    gen_c = st.columns(3, vertical_alignment="bottom")
    with gen_c[0]:
        if st.button("🔍 Business Overview (GenAI)"):
            st.session_state.selected_sponsor_name = selected_sponsor_name  
            pharma_gen.generate_sponsor_business_overview(selected_sponsor_name)    
    with gen_c[1]:
        if st.button("🔍 Comprehensive BI Report (GenAI) "):
            st.session_state.selected_sponsor_name = selected_sponsor_name
            pharma_gen.generate_sponsor_comprehensive_bi_report(selected_sponsor_name)

@st.cache_data(ttl=CACHE_TTL, show_spinner="🔍 Analyzing comprehensive sponsor profile...")
def get_comprehensive_sponsor_profile(sponsor_name: str) -> Dict:
    """Get comprehensive sponsor profile data from AACT database"""

    profile_data = {}
    param={'sponsor_name':sponsor_name}

    # 1. Basic Sponsor Information
    basic_sponsor_query = f"""
    SELECT 
        sp.name as sponsor_name,
        sp.agency_class,
        COUNT(DISTINCT sp.nct_id) as total_studies,
        COUNT(DISTINCT CASE WHEN sp.lead_or_collaborator = 'lead' THEN sp.nct_id END) as lead_studies,
        COUNT(DISTINCT CASE WHEN sp.lead_or_collaborator = 'collaborator' THEN sp.nct_id END) as collaborator_studies
    FROM sponsors sp
    WHERE sp.name = :sponsor_name
    GROUP BY sp.name, sp.agency_class
    """
    
    profile_data['basic_info'] = st.session_state.db_manager.execute_query(basic_sponsor_query,param)

    # 2. Study Status Distribution
    status_distribution_query = f"""
    SELECT 
        s.overall_status,
        COUNT(DISTINCT s.nct_id) as study_count,
        ROUND(100.0 * COUNT(DISTINCT s.nct_id) / NULLIF((SELECT COUNT(DISTINCT s2.nct_id) 
    FROM studies s2 
    JOIN sponsors sp2 ON s2.nct_id = sp2.nct_id 
    WHERE sp2.name = :sponsor_name),0), 2) as percentage
    FROM studies s
    JOIN sponsors sp ON s.nct_id = sp.nct_id
    WHERE sp.name = :sponsor_name
    GROUP BY s.overall_status
    ORDER BY study_count DESC
    """
    profile_data['status_distribution'] = st.session_state.db_manager.execute_query(status_distribution_query,param)

    # 3. Study Type Distribution
    type_distribution_query = f"""
    SELECT 
        s.study_type,
        COUNT(DISTINCT s.nct_id) as study_count,
        ROUND(100.0 * COUNT(DISTINCT s.nct_id) / NULLIF((SELECT COUNT(DISTINCT s2.nct_id) 
        FROM studies s2 
        JOIN sponsors sp2 ON s2.nct_id = sp2.nct_id 
        WHERE sp2.name = :sponsor_name),0), 2) as percentage
    FROM studies s
    JOIN sponsors sp ON s.nct_id = sp.nct_id
    WHERE sp.name = :sponsor_name
    GROUP BY s.study_type
    ORDER BY study_count DESC
    """
    profile_data['type_distribution'] = st.session_state.db_manager.execute_query(type_distribution_query,param)

    # 4. Phase Distribution
    phase_distribution_query = f"""
    SELECT 
        s.phase,
        COUNT(DISTINCT s.nct_id) as study_count,
        ROUND(100.0 * COUNT(DISTINCT s.nct_id) / NULLIF((SELECT COUNT(DISTINCT s2.nct_id) 
        FROM studies s2 
        JOIN sponsors sp2 ON s2.nct_id = sp2.nct_id 
        WHERE sp2.name = :sponsor_name AND s2.phase IS NOT NULL),0), 2) as percentage
    FROM studies s
    JOIN sponsors sp ON s.nct_id = sp.nct_id
    WHERE sp.name = :sponsor_name AND s.phase IS NOT NULL
    GROUP BY s.phase
    ORDER BY study_count DESC
    """
    profile_data['phase_distribution'] = st.session_state.db_manager.execute_query(phase_distribution_query,param)

    # 5. Geographic Reach
    geographic_reach_query = f"""
    SELECT 
        f.country,
        f.state,
        COUNT(DISTINCT f.nct_id) as studies_count,
        COUNT(DISTINCT f.name) as unique_facilities
    FROM facilities f
    JOIN sponsors sp ON f.nct_id = sp.nct_id
    WHERE sp.name = :sponsor_name
    GROUP BY f.country, f.state
    ORDER BY studies_count DESC
    """
    profile_data['geographic_reach'] = st.session_state.db_manager.execute_query(geographic_reach_query,param)

    # 6. Therapeutic Areas (Conditions)
    therapeutic_areas_query = f"""
    SELECT 
        c.name as condition_name,
        COUNT(DISTINCT c.nct_id) as studies_count,
        ROUND(100.0 * COUNT(DISTINCT c.nct_id) / NULLIF((SELECT COUNT(DISTINCT c2.nct_id) 
        FROM conditions c2 JOIN sponsors sp2 ON c2.nct_id = sp2.nct_id 
        WHERE sp2.name = :sponsor_name),0), 2) as percentage
    FROM conditions c
    JOIN sponsors sp ON c.nct_id = sp.nct_id
    WHERE sp.name = :sponsor_name
    GROUP BY c.name
    ORDER BY studies_count DESC
    LIMIT 20
    """
    profile_data['therapeutic_areas'] = st.session_state.db_manager.execute_query(therapeutic_areas_query,param)

    # 7. Intervention Types
    intervention_types_query = f"""
    SELECT 
        i.intervention_type,
        COUNT(DISTINCT i.nct_id) as studies_count,
        COUNT(*) as total_interventions
    FROM interventions i
    JOIN sponsors sp ON i.nct_id = sp.nct_id
    WHERE sp.name = :sponsor_name
    GROUP BY i.intervention_type
    ORDER BY studies_count DESC
    """
    profile_data['intervention_types'] = st.session_state.db_manager.execute_query(intervention_types_query,param)

    # 8. Timeline Analysis      
    timeline_analysis_query = f"""
    SELECT 
        EXTRACT(YEAR FROM s.start_date) as start_year,
        COUNT(DISTINCT s.nct_id) as studies_count,
        AVG(s.enrollment) as avg_enrollment,
        AVG(cv.actual_duration) as avg_duration
    FROM studies s
    JOIN sponsors sp ON s.nct_id = sp.nct_id
    LEFT JOIN calculated_values cv ON s.nct_id = cv.nct_id
    WHERE sp.name = :sponsor_name AND s.start_date IS NOT NULL
    GROUP BY EXTRACT(YEAR FROM s.start_date)
    ORDER BY start_year
    """
    profile_data['timeline_analysis'] = st.session_state.db_manager.execute_query(timeline_analysis_query,param)

    # 9. Success Metrics
    success_metrics_query = f"""
    SELECT 
        COUNT(DISTINCT s.nct_id) as total_studies,
        COUNT(DISTINCT CASE WHEN s.overall_status = 'COMPLETED' THEN s.nct_id END) as completed_studies,
        COUNT(DISTINCT CASE WHEN s.overall_status IN ('TERMINATED', 'SUSPENDED', 'WITHDRAWN') THEN s.nct_id END) as failed_studies,
        COUNT(DISTINCT CASE WHEN cv.were_results_reported = true THEN s.nct_id END) as results_reported,
        AVG(s.enrollment) as avg_enrollment,
        AVG(cv.actual_duration) as avg_duration,
        AVG(cv.months_to_report_results) as avg_months_to_report
    FROM studies s
    JOIN sponsors sp ON s.nct_id = sp.nct_id
    LEFT JOIN calculated_values cv ON s.nct_id = cv.nct_id
    WHERE sp.name = :sponsor_name
    """
    profile_data['success_metrics'] = st.session_state.db_manager.execute_query(success_metrics_query,param)

    # 10. Collaborations (Other Sponsors)
    collaborations_query = f"""
    SELECT 
        sp2.name as collaborator_name,
        sp2.agency_class as collaborator_class,
        COUNT(DISTINCT sp2.nct_id) as shared_studies
    FROM sponsors sp1
    JOIN sponsors sp2 ON sp1.nct_id = sp2.nct_id
    WHERE sp1.name = :sponsor_name 
    AND sp2.name != :sponsor_name
    GROUP BY sp2.name, sp2.agency_class
    ORDER BY shared_studies DESC
    LIMIT 15
    """
    profile_data['collaborations'] = st.session_state.db_manager.execute_query(collaborations_query,param)

    # 11. Key Investigators
    if st.session_state.database_type == 'sample':
        investigators_query = f"""
        SELECT 
            i.name as investigator_name,
            i.affiliation,
            COUNT(DISTINCT si.nct_id) as studies_count,
            COUNT(DISTINCT CASE WHEN si.role = 'PRINCIPAL INVESTIGATOR' THEN si.nct_id END) as studies_as_pi
        FROM investigators i
        JOIN study_investigators si ON i.investigator_id = si.investigator_id
        JOIN sponsors sp ON si.nct_id = sp.nct_id
        WHERE sp.name = :sponsor_name
        GROUP BY i.name, i.affiliation
        ORDER BY studies_count DESC
        LIMIT 10
        """
    else:
        investigators_query = f"""
        SELECT 
            oo.name as investigator_name,
            oo.affiliation,
            COUNT(DISTINCT oo.nct_id) as studies_count,
            COUNT(DISTINCT CASE WHEN oo.role = 'PRINCIPAL INVESTIGATOR' THEN oo.nct_id END) as studies_as_pi
        FROM overall_officials oo
        JOIN sponsors sp ON oo.nct_id = sp.nct_id
        WHERE sp.name = :sponsor_name AND oo.name IS NOT NULL
        GROUP BY oo.name, oo.affiliation
        ORDER BY studies_count DESC
        LIMIT 10
        """
    profile_data['key_investigators'] = st.session_state.db_manager.execute_query(investigators_query,param)

    # 12. Facilities Network
    facilities_network_query = f"""
    SELECT 
        f.name as facility_name,
        f.city,
        f.state,
        f.country,
        COUNT(DISTINCT f.nct_id) as studies_count
    FROM facilities f
    JOIN sponsors sp ON f.nct_id = sp.nct_id
    WHERE sp.name = :sponsor_name
    GROUP BY f.name, f.city, f.state, f.country
    ORDER BY studies_count DESC
    LIMIT 20
    """
    profile_data['facilities_network'] = st.session_state.db_manager.execute_query(facilities_network_query,param)

    # 13. Study Outcomes Focus
    outcomes_focus_query = f"""
    SELECT 
        o.outcome_type,
        COUNT(DISTINCT o.nct_id) as studies_count,
        COUNT(*) as total_outcomes
    FROM outcomes o
    JOIN sponsors sp ON o.nct_id = sp.nct_id
    WHERE sp.name = :sponsor_name
    GROUP BY o.outcome_type
    ORDER BY studies_count DESC
    """
    profile_data['outcomes_focus'] = st.session_state.db_manager.execute_query(outcomes_focus_query,param)

    # 14. Regulatory Focus
    if st.session_state.db_manager.database_type == 'sample':
        regulatory_focus_query = f"""
        SELECT 
            CASE 
                WHEN s.is_fda_regulated_drug = '1' THEN 'FDA Drug Regulated'
                WHEN s.is_fda_regulated_device = '1' THEN 'FDA Device Regulated'
                ELSE 'Other/Not Specified'
            END as regulatory_type,
            COUNT(DISTINCT s.nct_id) as studies_count
        FROM studies s
        JOIN sponsors sp ON s.nct_id = sp.nct_id
        WHERE sp.name = :sponsor_name
        GROUP BY regulatory_type
        ORDER BY studies_count DESC
        """
    else:
        regulatory_focus_query = f"""
        SELECT 
            CASE 
                WHEN s.is_fda_regulated_drug = true THEN 'FDA Drug Regulated'
                WHEN s.is_fda_regulated_device = true THEN 'FDA Device Regulated'
                ELSE 'Other/Not Specified'
            END as regulatory_type,
            COUNT(DISTINCT s.nct_id) as studies_count
        FROM studies s
        JOIN sponsors sp ON s.nct_id = sp.nct_id
        WHERE sp.name = :sponsor_name
        GROUP BY regulatory_type
        ORDER BY studies_count DESC
        """
    profile_data['regulatory_focus'] = st.session_state.db_manager.execute_query(regulatory_focus_query,param)

    return profile_data


def render_comprehensive_sponsor_profile(selected_sponsor_name):
    profile_data = get_comprehensive_sponsor_profile(selected_sponsor_name)
    
    if profile_data:
        # profile_data['success_metrics']
        # Profile Header
        st.markdown(f"""
        <div class="success-box">
        <h3>🏢 Comprehensive Sponsor Profile: {selected_sponsor_name}</h3>
        <p>Complete analysis of clinical trial portfolio and organizational capabilities</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Tab-based profile sections
        tab_names = [
            "📊 Portfolio Overview",
            "📈 Performance Metrics", 
            "🌍 Geographic Analysis",
            "🔬 Therapeutic Focus",
            "🤝 Collaborations",
            "👨‍⚕️ Key Investigators",
            "🏥 Facilities Network",
            "📊 Visualizations",
            "📑 Study Outcomes Focus"
        ]
        
        tabs = st.tabs(tab_names)
        
        # Tab 1: Portfolio Overview
        with tabs[0]:
            st.subheader("📊 Study Portfolio Overview")
            
            # Success metrics summary
            if not profile_data['success_metrics'].empty:
                success_data = profile_data['success_metrics'].iloc[0]
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Total Studies", int(success_data['total_studies']))
                
                with col2:
                    completion_rate = (success_data['completed_studies'] / success_data['total_studies']) * 100 if success_data['total_studies'] > 0 else 0
                    st.metric("Completion Rate", f"{completion_rate:.1f}%")
                
                with col3:
                    avg_enrollment = success_data['avg_enrollment'] or 0
                    st.metric("Avg Enrollment", f"{avg_enrollment:.0f}")
                
                with col4:
                    avg_duration = success_data['avg_duration'] or 0
                    st.metric("Avg Duration", f"{avg_duration:.1f} months")
                
                with col5:
                    reporting_rate = (success_data['results_reported'] / success_data['completed_studies']) * 100 if success_data['completed_studies'] > 0 else 0
                    st.metric("Results Reporting", f"{reporting_rate:.1f}%")
            
            # Study Status Distribution
            if not profile_data['status_distribution'].empty:
                st.subheader("📈 Study Status Breakdown")
                
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    status_df = profile_data['status_distribution'].copy()
                    status_df.columns = ['Status', 'Count', 'Percentage']
                    st.dataframe(status_df, use_container_width=True, hide_index=True)
                
                with col2:
                    # Create simple metrics
                    c=st.columns(3,gap="small",vertical_alignment="top")
                    for i, row in profile_data['status_distribution'].iterrows():
                        c[i%3].metric(row['overall_status'], row['study_count'], f"{row['percentage']}%")
            
            # Study Type Distribution
            if not profile_data['type_distribution'].empty:
                st.subheader("🔬 Study Type Analysis")
                type_df = profile_data['type_distribution'].copy()
                type_df.columns = ['Study Type', 'Count', 'Percentage']
                st.dataframe(type_df, use_container_width=True, hide_index=True)
            
            # Phase Distribution
            if not profile_data['phase_distribution'].empty:
                st.subheader("⚗️ Clinical Phase Distribution")
                phase_df = profile_data['phase_distribution'].copy()
                phase_df.columns = ['Phase', 'Count', 'Percentage']
                st.dataframe(phase_df, use_container_width=True, hide_index=True)
        
        # Additional tabs with content similar to above...
        # (Implementation continues with all 8 tabs)

        with tabs[1]:
            st.subheader("📈 Performance Metrics")
            if not profile_data['success_metrics'].empty:
                metrics = profile_data['success_metrics'].iloc[0]
                col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
                with col1:
                    st.metric("Total Studies", int(metrics['total_studies']))
                with col2:
                    st.metric("Completed", int(metrics['completed_studies']))
                with col3:
                    st.metric("Failed", int(metrics['failed_studies']))
                with col4:
                    st.metric("Results Reported", int(metrics['results_reported']))
                with col5:
                    completion_rate = (metrics['completed_studies'] / metrics['total_studies'] * 100) if metrics['total_studies'] else 0
                    st.metric("Completion Rate", f"{completion_rate:.1f}%")
                with col6:
                    avg_enrollment = metrics['avg_enrollment'] if metrics['avg_enrollment'] is not None else 0
                    st.metric("Avg Enrollment", f"{avg_enrollment:.0f}")
                with col7:
                    avg_duration = metrics['avg_duration'] if metrics['avg_duration'] is not None else 0
                    st.metric("Avg Duration (months)", f"{avg_duration:.1f}")

                # Results reporting rate
                reporting_rate = (metrics['results_reported'] / metrics['completed_studies'] * 100) if metrics['completed_studies'] else 0
                st.caption(f"Results Reporting Rate: {reporting_rate:.1f}% of completed studies")

                # Optionally show months to report results
                if 'avg_months_to_report' in metrics and metrics['avg_months_to_report'] is not None:
                    st.caption(f"Avg Months to Report Results: {metrics['avg_months_to_report']:.1f}")

            else:
                st.info("No performance metrics available for this sponsor.")

        # 2. 🌍 Geographic Analysis
        with tabs[2]:
            st.subheader("🌍 Geographic Analysis")
            if not profile_data['facilities_network'].empty:
                facilities_df = profile_data['facilities_network'].copy()
                facilities_df.columns = ['Facility Name', 'City', 'State', 'Country', 'Studies Count']
                st.dataframe(facilities_df, use_container_width=True, hide_index=True)
                # Optionally, show a map if lat/lon available
                if {'lat', 'lon'}.issubset(facilities_df.columns):
                    st.map(facilities_df.rename(columns={'lat': 'latitude', 'lon': 'longitude'}))
            else:
                st.info("No facility network data available for this sponsor.")

        # 3. 🔬 Therapeutic Focus
        with tabs[3]:
            st.subheader("🔬 Therapeutic Focus")
            if not profile_data['therapeutic_areas'].empty:
                ta_df = profile_data['therapeutic_areas'].copy()
                ta_df.columns = ['Condition', 'Studies Count', 'Percentage']
                st.dataframe(ta_df, use_container_width=True, hide_index=True)
            else:
                st.info("No therapeutic area data available.")

        # 4. 🤝 Collaborations
        with tabs[4]:
            st.subheader("🤝 Collaborations")
            if not profile_data['collaborations'].empty:
                collab_df = profile_data['collaborations'].copy()
                collab_df.columns = ['Collaborator Name', 'Collaborator Class', 'Shared Studies']
                st.dataframe(collab_df, use_container_width=True, hide_index=True)
            else:
                st.info("No collaboration data available.")

        # 5. 👨‍⚕️ Key Investigators
        with tabs[5]:
            st.subheader("👨‍⚕️ Key Investigators")
            if not profile_data['key_investigators'].empty:
                inv_df = profile_data['key_investigators'].copy()
                inv_df.columns = ['Investigator Name', 'Affiliation', 'Studies Count', 'Studies as PI']
                st.dataframe(inv_df, use_container_width=True, hide_index=True)
            else:
                st.info("No key investigator data available.")

        # 6. 🏥 Facilities Network
        with tabs[6]:
            st.subheader("🏥 Facilities Network")
            if not profile_data['facilities_network'].empty:
                fac_df = profile_data['facilities_network'].copy()
                fac_df.columns = ['Facility Name', 'City', 'State', 'Country', 'Studies Count']
                st.dataframe(fac_df, use_container_width=True, hide_index=True)
            else:
                st.info("No facilities network data available.")

        # 7. 📊 Visualizations
        with tabs[7]:
            st.subheader("📊 Visualizations")
            # Example: Timeline Analysis
            if not profile_data['timeline_analysis'].empty:
                timeline_df = profile_data['timeline_analysis'].copy()
                timeline_df.columns = ['Start Year', 'Studies Count', 'Avg Enrollment', 'Avg Duration']
                # Create separate line charts for each metric to avoid mixed types
               
                st.subheader("Studies Count Over Time")
                tg,td=st.tabs(['Graph','Table'])
                with tg:
                    st.plotly_chart({
                        "data": [{
                            "x": timeline_df['Start Year'],
                            "y": timeline_df['Studies Count'],
                            "type": "scatter",
                            "mode": "lines+markers",
                            "name": "Studies Count"
                        }],
                        "layout": {
                            "title": "Studies Count Over Time",
                            "xaxis": {"title": "Start Year"},
                            "yaxis": {"title": "Number of Studies"}
                        }
                    })
                with td:
                    st.dataframe(timeline_df.set_index('Start Year')['Studies Count'])
                
                if not timeline_df.set_index('Start Year')['Avg Enrollment'].empty:
                    st.subheader("Average Enrollment Over Time") 
                    tg,td=st.tabs(['Graph','Table'])
                    with tg:
                        st.plotly_chart({
                            "data": [{
                                "x": timeline_df['Start Year'],
                                "y": timeline_df['Avg Enrollment'], 
                                "type": "scatter",
                                "mode": "lines+markers",
                                "name": "Average Enrollment"
                            }],
                            "layout": {
                                "title": "Average Enrollment Over Time",
                                "xaxis": {"title": "Start Year"},
                                "yaxis": {"title": "Average Enrollment"}
                            }
                        })
                    with td:
                        st.dataframe(timeline_df.set_index('Start Year')['Avg Enrollment'])
                if not timeline_df.set_index('Start Year')['Avg Duration'].empty: 
                    st.subheader("Average Duration Over Time")
                    tg,td=st.tabs(['Graph','Table'])
                    with tg:
                        st.plotly_chart({
                            "data": [{
                                "x": timeline_df['Start Year'],
                                "y": timeline_df['Avg Duration'],
                                "type": "scatter",
                                "mode": "lines+markers",
                                "name": "Average Duration"
                            }],
                            "layout": {
                                "title": "Average Study Duration Over Time",
                                "xaxis": {"title": "Start Year"},
                                "yaxis": {"title": "Average Duration (Days)"}
                            }
                        })
                    with td:
                        st.dataframe(timeline_df.set_index('Start Year')['Avg Duration'])
            else:
                st.info("No timeline analysis data available.")

            # Example: Phase Distribution Pie Chart
            if not profile_data['phase_distribution'].empty:
                phase_df = profile_data['phase_distribution'].copy()
                phase_df.columns = ['Phase', 'Count', 'Percentage']
                tg,td=st.tabs(['Graph','Table'])
                with tg:
                    st.plotly_chart(
                        {
                            "data": [{
                                "labels": phase_df['Phase'],
                                "values": phase_df['Count'],
                                "type": "pie"
                            }],
                            "layout": {"title": "Phase Distribution"}
                            }
                        )
                with td:
                    st.dataframe(phase_df)
            # Add more visualizations as needed

        # 8. 📑 Outcomes Focus
        with tabs[8]:
            st.subheader("📑 Study Outcomes Focus")
            if not profile_data['outcomes_focus'].empty:
                out_df = profile_data['outcomes_focus'].copy()
                out_df.columns = ['Outcome Type', 'Studies Count', 'Total Outcomes']
                st.dataframe(out_df, use_container_width=True, hide_index=True)
            else:
                st.info("No outcomes focus data available.")
        
        # Export functionality
        st.markdown("---")
        st.subheader("📥 Export Sponsor Profile")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Download Profile Summary"):
                summary_text = f"""
                # Sponsor Profile: {selected_sponsor_name}
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                ## Key Metrics
                - Total Studies: {profile_data['success_metrics'].iloc[0]['total_studies'] if not profile_data['success_metrics'].empty else 'N/A'}
                - Completed Studies: {profile_data['success_metrics'].iloc[0]['completed_studies'] if not profile_data['success_metrics'].empty else 'N/A'}
                - Average Enrollment: {profile_data['success_metrics'].iloc[0]['avg_enrollment']:.0f if not profile_data['success_metrics'].empty and profile_data['success_metrics'].iloc[0]['avg_enrollment'] else 'N/A'}
                """
                
                st.download_button(
                    label="⬇️ Download Summary",
                    data=summary_text,
                    file_name=f"sponsor_profile_{selected_sponsor_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )
    
    else:
        st.error("Failed to generate sponsor profile. Please try a different sponsor.")

@st.fragment
def sponsor_profile_comprehensive_section(selected_sponsor_name):
    col1, col2 = st.columns([3, 1])
    
    with col1:
        generate_sponsor_profile=st.button("🔍 Generate Comprehensive Sponsor Profile", type="primary")
    
    with col2:
        include_visualizations = st.checkbox("Include Visualizations", value=True)
    
    # Display comprehensive profile
    if generate_sponsor_profile:
        render_comprehensive_sponsor_profile(selected_sponsor_name)
    
    elif not st.session_state.get('generate_sponsor_profile', False):
        st.info("👆 Click 'Generate Comprehensive Sponsor Profile' to analyze detailed sponsor metrics.")
        


def render_sponsor_profiles_page():
    """Render the sponsor profiles page"""
    
    st.header("🏢 Sponsor Profile Analysis",divider=True)
     # Check if database is connected
    if not check_database_connection():
        return
        
        
    st.markdown("""
    <div class="insight-box">
    <h5>🔍 Strategic Sponsor Insights</h5>
    <p>Analyze sponsor profiles to understand their clinical trial strategies and market trends.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 1: Sponsor Selection
    st.subheader("🔍 Sponsor Selection")
    col1, col2 = st.columns([1, 1],vertical_alignment="bottom")
    with col1:
        selected_sponsor_name, sponsor_info,sp_stock_df = sponsor_selection_widget()
    

    if not selected_sponsor_name:
        st.info("👆 Click 'Select Sponsor' to view their studies.")
        return

    if selected_sponsor_name:
        # Quick Overview
        render_sponsor_quick_overview(selected_sponsor_name, sponsor_info, sp_stock_df)
        st.markdown("---")
        
        # Generate Comprehensive Profile
        sponsor_profile_comprehensive_section(selected_sponsor_name)


render_sponsor_profiles_page()