"""
Study Reports Page
=================

Detailed clinical study reports and analysis.
"""

import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List
import plotly.express as px
import plotly.graph_objects as go
from utils.common_service import get_sponsors_list, get_studies_by_sponsor,check_database_connection

CACHE_TTL = 60 * 60 * 24  # 24 hours


# @st.cache_data(ttl=CACHE_TTL,show_spinner="Loading sponsors list...")
# def get_sponsors_list():
#     query=st.session_state.config.get_query('study_reports', 'sponsors_list')
#     return st.session_state.db_manager.execute_query(query)

# @st.cache_data(ttl=CACHE_TTL, show_spinner="Loading studies by sponsor...")   
# def get_studies_by_sponsor(sponsor_name):
#     query = st.session_state.config.get_query('study_reports', 'studies_by_sponsor')
#     # Ensure params is a dictionary if using SQLAlchemy/text, or a tuple for SQLite
#     db_type = getattr(st.session_state.db_manager, "database_type", None)
#     if db_type == "real":
#         # SQLAlchemy/text expects a dictionary for named parameters, but our query uses %s, so use tuple
#         params = {'sponsor_name':sponsor_name}
#     else:
#         params = (sponsor_name,)
#     return st.session_state.db_manager.execute_query(query, params)

@st.cache_data(ttl=CACHE_TTL,show_spinner="Loading detailed study report...")
def get_detailed_study_report(nct_id):
    """Get comprehensive study report data"""
    report_data = {}
    
    # 1. Basic Study Information
    basic_info_query = f"""
    SELECT 
        s.*,
        cv.number_of_facilities,
        cv.number_of_nsae_subjects,
        cv.number_of_sae_subjects,
        cv.were_results_reported,
        cv.months_to_report_results,
        cv.has_us_facility,
        cv.minimum_age_num,
        cv.maximum_age_num,
        cv.actual_duration
    FROM studies s
    LEFT JOIN calculated_values cv ON s.nct_id = cv.nct_id
    WHERE s.nct_id = '{nct_id}'
    """
    report_data['basic_info'] = st.session_state.db_manager.execute_query(basic_info_query)
    
    # 2. Sponsors Information
    sponsors_query = f"""
    SELECT 
        sp.name,
        sp.agency_class,
        sp.lead_or_collaborator
    FROM sponsors sp
    WHERE sp.nct_id = '{nct_id}'
    ORDER BY sp.lead_or_collaborator, sp.name
    """
    report_data['sponsors'] = st.session_state.db_manager.execute_query(sponsors_query)
    
    # 3. Facilities and Sites
    facilities_query = f"""
    SELECT 
        f.name as facility_name,
        f.city,
        f.state,
        f.country,
        f.status
    FROM facilities f
    WHERE f.nct_id = '{nct_id}'
    ORDER BY f.country, f.state, f.city
    """
    report_data['facilities'] = st.session_state.db_manager.execute_query(facilities_query)
    
    # 4. Conditions
    conditions_query = f"""
    SELECT 
        c.name as condition_name
    FROM conditions c
    WHERE c.nct_id = '{nct_id}'
    ORDER BY c.name
    """
    report_data['conditions'] = st.session_state.db_manager.execute_query(conditions_query)
    
    # 5. Interventions
    interventions_query = f"""
    SELECT 
        i.intervention_type,
        i.name as intervention_name,
        i.description
    FROM interventions i
    WHERE i.nct_id = '{nct_id}'
    ORDER BY i.intervention_type, i.name
    """
    report_data['interventions'] = st.session_state.db_manager.execute_query(interventions_query)
    
    # 6. Primary and Secondary Outcomes
    outcomes_query = f"""
    SELECT 
        o.outcome_type,
        o.title,
        o.description,
        o.time_frame,
        o.units
    FROM outcomes o
    WHERE o.nct_id = '{nct_id}'
    ORDER BY 
        CASE WHEN o.outcome_type = 'Primary' THEN 1 
                WHEN o.outcome_type = 'Secondary' THEN 2 
                ELSE 3 END,
        o.title
    """
    report_data['outcomes'] = st.session_state.db_manager.execute_query(outcomes_query)
    
    # 7. Eligibility Criteria
    eligibility_query = f"""
    SELECT 
        e.gender,
        e.minimum_age,
        e.maximum_age,
        e.healthy_volunteers,
        e.criteria,
        e.adult,
        e.child,
        e.older_adult
    FROM eligibilities e
    WHERE e.nct_id = '{nct_id}'
    """
    report_data['eligibility'] = st.session_state.db_manager.execute_query(eligibility_query)
    
    # 8. Study Design
    design_query = f"""
    SELECT 
        d.allocation,
        d.intervention_model,
        d.primary_purpose,
        d.masking,
        d.subject_masked,
        d.caregiver_masked,
        d.investigator_masked,
        d.outcomes_assessor_masked
    FROM designs d
    WHERE d.nct_id = '{nct_id}'
    """
    report_data['design'] = st.session_state.db_manager.execute_query(design_query)
    
    # 9. Principal Investigators and Officials
    if st.session_state.database_type == 'sample':
        investigators_query = f"""
        SELECT 
            i.name,
            i.affiliation,
            i.degree,
            si.role
        FROM investigators i
        JOIN study_investigators si ON i.investigator_id = si.investigator_id
        WHERE si.nct_id = '{nct_id}'
        ORDER BY 
            CASE WHEN si.role = 'Principal Investigator' THEN 1 ELSE 2 END,
            i.name
        """
    else:
        investigators_query = f"""
        SELECT 
            oo.name,
            oo.affiliation,
            oo.role
        FROM overall_officials oo
        WHERE oo.nct_id = '{nct_id}'
        ORDER BY 
            CASE WHEN oo.role = 'Principal Investigator' THEN 1 ELSE 2 END,
            oo.name
        """
    report_data['investigators'] = st.session_state.db_manager.execute_query(investigators_query)
    
    # 10. Responsible Party
    responsible_party_query = f"""
    SELECT 
        rp.responsible_party_type,
        rp.name,
        rp.organization
    FROM responsible_parties rp
    WHERE rp.nct_id = '{nct_id}'
    """
    report_data['responsible_party'] = st.session_state.db_manager.execute_query(responsible_party_query)
    
    # 11. Reported Events (Adverse Events)
    if st.session_state.database_type == 'real':
        adverse_events_query = f"""
        SELECT 
            re.event_type,
            re.default_vocab,
            re.default_assessment,
            re.subjects_affected,
            re.subjects_at_risk,
            re.description,
            re.event_count,
            re.organ_system,
            re.adverse_event_term
        FROM reported_events re
        WHERE re.nct_id = '{nct_id}'
        ORDER BY re.event_type, re.subjects_affected DESC
        """
        report_data['adverse_events'] = st.session_state.db_manager.execute_query(adverse_events_query)
    else:
        # Sample data - create placeholder
        report_data['adverse_events'] = pd.DataFrame()
    
    # 12. Study References (Publications)
    if st.session_state.database_type == 'real':
        references_query = f"""
        SELECT 
            sr.pmid,
            sr.reference_type,
            sr.citation
        FROM study_references sr
        WHERE sr.nct_id = '{nct_id}'
        ORDER BY sr.reference_type, sr.pmid
        """
        report_data['references'] = st.session_state.db_manager.execute_query(references_query)
    else:
        # Sample data - create placeholder
        report_data['references'] = pd.DataFrame()
    
    # 13. Milestones (if available)
    if st.session_state.database_type == 'real':
        milestones_query = f"""
        SELECT 
            m.title,
            m.period,
            m.description,
            m.count
        FROM milestones m
        WHERE m.nct_id = '{nct_id}'
        ORDER BY m.period, m.title
        """
        report_data['milestones'] = st.session_state.db_manager.execute_query(milestones_query)
    else:
        # Sample data - create placeholder
        report_data['milestones'] = pd.DataFrame()
    
    # 14. Baseline Measurements (if available)
    if st.session_state.database_type == 'real':
        baseline_query = f"""
        SELECT 
            bm.title,
            bm.description,
            bm.units,
            bm.param_type,
            bm.param_value,
            bm.dispersion_type,
            bm.dispersion_value,
            bm.category
        FROM baseline_measurements bm
        WHERE bm.nct_id = '{nct_id}'
        ORDER BY bm.title, bm.category
        """
        report_data['baseline_measurements'] = st.session_state.db_manager.execute_query(baseline_query)
    else:
        # Sample data - create placeholder
        report_data['baseline_measurements'] = pd.DataFrame()
    
    # 15. Outcome Measurements (Results)
    if st.session_state.database_type == 'real':
        outcome_measurements_query = f"""
        SELECT 
            om.title,
            om.description,
            om.units,
            om.param_type,
            om.param_value,
            om.dispersion_type,
            om.dispersion_value,
            om.category
        FROM outcome_measurements om
        WHERE om.nct_id = '{nct_id}'
        ORDER BY om.title, om.category
        """
        report_data['outcome_measurements'] = st.session_state.db_manager.execute_query(outcome_measurements_query)
    else:
        # Sample data - create placeholder
        report_data['outcome_measurements'] = pd.DataFrame()
    
    return report_data

@st.fragment
def create_study_report_visualizations(report_data: Dict, nct_id: str) -> List:
    """Create visualizations for study report"""
    
    figures = []
    
    # 1. Study Timeline Visualization
    if not report_data['basic_info'].empty:
        basic_info = report_data['basic_info'].iloc[0]
        
        # Create timeline chart
        timeline_data = []
        if pd.notna(basic_info.get('start_date')):
            timeline_data.append({
                'Task': 'Study Period',
                'Start': basic_info['start_date'],
                'Finish': basic_info.get('completion_date', basic_info.get('primary_completion_date', basic_info['start_date'])),
                'Type': 'Study Duration'
            })
        
        if timeline_data:
            fig_timeline = px.timeline(
                timeline_data,
                x_start='Start',
                x_end='Finish',
                y='Task',
                color='Type',
                title=f'Study Timeline - {nct_id}'
            )
            fig_timeline.update_layout(height=200)
            figures.append(('Study Timeline', fig_timeline))
    
    # 2. Geographic Distribution of Sites
    if not report_data['facilities'].empty:
        facilities = report_data['facilities']
        country_counts = facilities['country'].value_counts()
        
        if len(country_counts) > 1:
            fig_geo = px.pie(
                values=country_counts.values,
                names=country_counts.index,
                title='Geographic Distribution of Study Sites'
            )
            figures.append(('Site Distribution', fig_geo))
        
        # Sites by country bar chart
        if len(facilities) > 0:
            fig_sites = px.bar(
                x=country_counts.index,
                y=country_counts.values,
                title='Number of Sites by Country',
                labels={'x': 'Country', 'y': 'Number of Sites'}
            )
            fig_sites.update_layout(height=400)
            figures.append(('Sites by Country', fig_sites))
    
    # 3. Intervention Types
    if not report_data['interventions'].empty:
        interventions = report_data['interventions']
        intervention_counts = interventions['intervention_type'].value_counts()
        
        fig_interventions = px.pie(
            values=intervention_counts.values,
            names=intervention_counts.index,
            title='Intervention Types'
        )
        figures.append(('Intervention Types', fig_interventions))
    
    # 4. Adverse Events Analysis (if available)
    if not report_data['adverse_events'].empty:
        adverse_events = report_data['adverse_events']
        
        # Events by type
        if 'event_type' in adverse_events.columns:
            event_type_counts = adverse_events['event_type'].value_counts()
            fig_ae_types = px.bar(
                x=event_type_counts.index,
                y=event_type_counts.values,
                title='Adverse Events by Type',
                labels={'x': 'Event Type', 'y': 'Number of Events'}
            )
            figures.append(('Adverse Events by Type', fig_ae_types))
        
        # Severity analysis
        if 'subjects_affected' in adverse_events.columns:
            fig_ae_severity = px.scatter(
                adverse_events,
                x='subjects_at_risk',
                y='subjects_affected',
                title='Adverse Events: Subjects Affected vs At Risk',
                hover_data=['adverse_event_term'] if 'adverse_event_term' in adverse_events.columns else None
            )
            figures.append(('AE Severity Analysis', fig_ae_severity))
    
    # 5. Study Milestones (if available)
    if not report_data['milestones'].empty:
        milestones = report_data['milestones']
        milestone_counts = milestones.groupby('period')['count'].sum().reset_index()
        
        fig_milestones = px.bar(
            milestone_counts,
            x='period',
            y='count',
            title='Study Milestones by Period'
        )
        figures.append(('Study Milestones', fig_milestones))
    
    return figures

@st.fragment
def generate_study_report_summary(report_data: Dict, nct_id: str) -> str:
    """Generate a comprehensive study report summary"""
    
    if report_data['basic_info'].empty:
        return "No study information available."
    
    basic_info = report_data['basic_info'].iloc[0]
    
    summary = f"""
    # 📄 Clinical Study Report: {nct_id}
    
    ## 📋 Study Overview
    **Title:** {basic_info.get('brief_title', 'N/A')}
    
    **Official Title:** {basic_info.get('official_title', 'N/A')}
    
    **Status:** {basic_info.get('overall_status', 'N/A')}
    
    **Phase:** {basic_info.get('phase', 'N/A')}
    
    **Study Type:** {basic_info.get('study_type', 'N/A')}
    
    ## 📅 Timeline
    **Start Date:** {basic_info.get('start_date', 'N/A')}
    
    **Completion Date:** {basic_info.get('completion_date', 'N/A')}
    
    **Actual Duration:** {basic_info.get('actual_duration', 'N/A')} months
    
    ## 👥 Enrollment
    **Target Enrollment:** {basic_info.get('enrollment', 'N/A')} participants
    
    ## 🏥 Study Sites
    **Number of Facilities:** {basic_info.get('number_of_facilities', 'N/A')}
    
    **US Facility:** {'Yes' if basic_info.get('has_us_facility') else 'No'}
    
    ## 📊 Results Status
    **Results Reported:** {'Yes' if basic_info.get('were_results_reported') else 'No'}
    
    **Months to Report Results:** {basic_info.get('months_to_report_results', 'N/A')}
    
    ## ⚠️ Safety Information
    **Non-Serious Adverse Events:** {basic_info.get('number_of_nsae_subjects', 'N/A')} subjects
    
    **Serious Adverse Events:** {basic_info.get('number_of_sae_subjects', 'N/A')} subjects
    
    ## 👥 Age Range
    **Minimum Age:** {basic_info.get('minimum_age_num', 'N/A')} years
    
    **Maximum Age:** {basic_info.get('maximum_age_num', 'N/A')} years
    """
    
    return summary

@st.fragment
def comprehensive_report_section(selected_nct_id):

    st.markdown("---")
    
    # Report generation controls
    col1, col2, col3 = st.columns([2, 1, 1],vertical_alignment='center')
    
    with col1:
        if st.button("📋 Generate Comprehensive Report", type="primary"):
            st.session_state.generate_report = True
            st.session_state.selected_nct_id = selected_nct_id
    
    with col2:
        include_visualizations = st.checkbox("Include Visualizations", value=True)
    
    with col3:
        report_format = st.selectbox("Report Format", ["Detailed", "Summary", "Executive"])
    
    # Generate and display report
    if st.session_state.get('generate_report', False) and st.session_state.get('selected_nct_id') == selected_nct_id:
        
        
        report_data = get_detailed_study_report(selected_nct_id) 
        
        if report_data and not report_data['basic_info'].empty:
            
            # Report Header
            basic_info = report_data['basic_info'].iloc[0]
            st.markdown(f"""
            <div class="success-box">
            <h3>📄 Clinical Study Report: {selected_nct_id}</h3>
            <h4>{basic_info.get('brief_title', 'N/A')}</h4>
            <p><strong>Status:</strong> {basic_info.get('overall_status', 'N/A')} | 
            <strong>Phase:</strong> {basic_info.get('phase', 'N/A')} | 
            <strong>Type:</strong> {basic_info.get('study_type', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Tab-based report sections
            tab_names = [
                "📋 Study Overview", 
                "🏥 Sites & Facilities", 
                "👨‍⚕️ Investigators", 
                "💊 Interventions & Outcomes",
                "👥 Eligibility & Design",
                "⚠️ Safety & Events",
                "📊 Results & Publications",
                "📈 Visualizations"
            ]
            
            tabs = st.tabs(tab_names)
            
            # Tab 1: Study Overview
            with tabs[0]:
                st.subheader("📋 Basic Study Information")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **NCT ID:** {basic_info.get('nct_id', 'N/A')}  
                    **Brief Title:** {basic_info.get('brief_title', 'N/A')}  
                    **Official Title:** {basic_info.get('official_title', 'N/A')}  
                    **Overall Status:** {basic_info.get('overall_status', 'N/A')}  
                    **Phase:** {basic_info.get('phase', 'N/A')}  
                    **Study Type:** {basic_info.get('study_type', 'N/A')}  
                    **Source Class:** {basic_info.get('source_class', 'N/A')}
                    """)
                
                with col2:
                    st.markdown(f"""
                    **Start Date:** {basic_info.get('start_date', 'N/A')}  
                    **Completion Date:** {basic_info.get('completion_date', 'N/A')}  
                    **Primary Completion Date:** {basic_info.get('primary_completion_date', 'N/A')}  
                    **Actual Duration:** {basic_info.get('actual_duration', 'N/A')} months  
                    **Enrollment:** {basic_info.get('enrollment', 'N/A')} participants  
                    **Number of Arms:** {basic_info.get('number_of_arms', 'N/A')}  
                    **Has DMC:** {'Yes' if basic_info.get('has_dmc') else 'No'}
                    """)
                
                # Sponsors section
                if not report_data['sponsors'].empty:
                    st.subheader("💰 Sponsors")
                    sponsors_display = report_data['sponsors'].copy()
                    sponsors_display.columns = ['Sponsor Name', 'Agency Class', 'Role']
                    st.dataframe(sponsors_display, use_container_width=True, hide_index=True)
                
                # Conditions section
                if not report_data['conditions'].empty:
                    st.subheader("🔬 Medical Conditions")
                    conditions_list = report_data['conditions']['condition_name'].tolist()
                    st.write(", ".join(conditions_list))
            
            # Tab 2: Sites & Facilities
            with tabs[1]:
                st.subheader("🏥 Study Sites and Facilities")
                
                if not report_data['facilities'].empty:
                    facilities_df = report_data['facilities'].copy()
                    facilities_df.columns = ['Facility Name', 'City', 'State', 'Country', 'Status']
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Sites", len(facilities_df))
                    
                    with col2:
                        countries = facilities_df['Country'].nunique()
                        st.metric("Countries", countries)
                    
                    with col3:
                        us_sites = len(facilities_df[facilities_df['Country'] == 'United States'])
                        st.metric("US Sites", us_sites)
                    
                    with col4:
                        active_sites = len(facilities_df[facilities_df['Status'].isin(['Recruiting', 'Active, not recruiting'])])
                        st.metric("Active Sites", active_sites)
                    
                    # Facilities table
                    st.dataframe(facilities_df, use_container_width=True, hide_index=True)
                    
                    # Geographic breakdown
                    country_counts = facilities_df['Country'].value_counts()
                    st.subheader("🌍 Geographic Distribution")
                    for country, count in country_counts.items():
                        st.write(f"**{country}:** {count} sites")
                else:
                    st.info("No facility information available for this study.")
            
            # Tab 3: Investigators
            with tabs[2]:
                st.subheader("👨‍⚕️ Principal Investigators and Study Officials")
                
                if not report_data['investigators'].empty:
                    investigators_df = report_data['investigators'].copy()
                    
                    # Count by role
                    if 'role' in investigators_df.columns:
                        role_counts = investigators_df['role'].value_counts()
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            pi_count = role_counts.get('Principal Investigator', 0)
                            st.metric("Principal Investigators", pi_count)
                        
                        with col2:
                            sub_count = role_counts.get('Sub-Investigator', 0)
                            st.metric("Sub-Investigators", sub_count)
                        
                        with col3:
                            other_count = len(investigators_df) - pi_count - sub_count
                            st.metric("Other Officials", other_count)
                    
                    # Investigators table
                    display_cols = ['Name', 'Affiliation', 'Role']
                    if 'degree' in investigators_df.columns:
                        display_cols.insert(1, 'Degree')
                        investigators_df.columns = ['Name', 'Affiliation', 'Degree', 'Role']
                    else:
                        investigators_df.columns = display_cols
                    
                    st.dataframe(investigators_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No investigator information available for this study.")
                
                # Responsible Party
                if not report_data['responsible_party'].empty:
                    st.subheader("📋 Responsible Party")
                    resp_party = report_data['responsible_party'].iloc[0]
                    st.markdown(f"""
                    **Type:** {resp_party.get('responsible_party_type', 'N/A')}  
                    **Name:** {resp_party.get('name', 'N/A')}  
                    **Organization:** {resp_party.get('organization', 'N/A')}
                    """)
            
            # Tab 4: Interventions & Outcomes
            with tabs[3]:
                st.subheader("💊 Study Interventions")
                
                if not report_data['interventions'].empty:
                    interventions_df = report_data['interventions'].copy()
                    interventions_df.columns = ['Type', 'Name', 'Description']
                    
                    # Intervention type summary
                    type_counts = interventions_df['Type'].value_counts()
                    st.write("**Intervention Types:**")
                    for int_type, count in type_counts.items():
                        st.write(f"• {int_type}: {count}")
                    
                    st.dataframe(interventions_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No intervention information available.")
                
                st.subheader("🎯 Study Outcomes")
                
                if not report_data['outcomes'].empty:
                    outcomes_df = report_data['outcomes'].copy()
                    outcomes_df.columns = ['Type', 'Title', 'Description', 'Time Frame', 'Units']
                    
                    # Separate primary and secondary outcomes
                    primary_outcomes = outcomes_df[outcomes_df['Type'] == 'Primary']
                    secondary_outcomes = outcomes_df[outcomes_df['Type'] == 'Secondary']
                    
                    if not primary_outcomes.empty:
                        st.write("**Primary Outcomes:**")
                        st.dataframe(primary_outcomes, use_container_width=True, hide_index=True)
                    
                    if not secondary_outcomes.empty:
                        st.write("**Secondary Outcomes:**")
                        st.dataframe(secondary_outcomes, use_container_width=True, hide_index=True)
                else:
                    st.info("No outcome information available.")
            
            # Tab 5: Eligibility & Design
            with tabs[4]:
                st.subheader("👥 Eligibility Criteria")
                
                if not report_data['eligibility'].empty:
                    eligibility = report_data['eligibility'].iloc[0]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        **Gender:** {eligibility.get('gender', 'N/A')}  
                        **Minimum Age:** {eligibility.get('minimum_age', 'N/A')}  
                        **Maximum Age:** {eligibility.get('maximum_age', 'N/A')}  
                        **Healthy Volunteers:** {'Yes' if eligibility.get('healthy_volunteers') else 'No'}
                        """)
                    
                    with col2:
                        age_groups = []
                        if eligibility.get('adult'): age_groups.append('Adults')
                        if eligibility.get('child'): age_groups.append('Children')
                        if eligibility.get('older_adult'): age_groups.append('Older Adults')
                        
                        st.markdown(f"""
                        **Age Groups:** {', '.join(age_groups) if age_groups else 'N/A'}
                        """)
                    
                    if eligibility.get('criteria'):
                        st.subheader("📋 Detailed Criteria")
                        st.text_area("Inclusion/Exclusion Criteria", 
                                    eligibility['criteria'], height=200, disabled=True)
                
                st.subheader("🔬 Study Design")
                
                if not report_data['design'].empty:
                    design = report_data['design'].iloc[0]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        **Allocation:** {design.get('allocation', 'N/A')}  
                        **Intervention Model:** {design.get('intervention_model', 'N/A')}  
                        **Primary Purpose:** {design.get('primary_purpose', 'N/A')}  
                        **Masking:** {design.get('masking', 'N/A')}
                        """)
                    
                    with col2:
                        masking_details = []
                        if design.get('subject_masked'): masking_details.append('Subject')
                        if design.get('caregiver_masked'): masking_details.append('Caregiver')
                        if design.get('investigator_masked'): masking_details.append('Investigator')
                        if design.get('outcomes_assessor_masked'): masking_details.append('Outcomes Assessor')
                        
                        st.markdown(f"""
                        **Masked Parties:** {', '.join(masking_details) if masking_details else 'None specified'}
                        """)
            
            # Tab 6: Safety & Events
            with tabs[5]:
                st.subheader("⚠️ Safety Information")
                
                # Basic safety metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    nsae_count = basic_info.get('number_of_nsae_subjects', 0)
                    st.metric("Non-Serious AE Subjects", nsae_count)
                
                with col2:
                    sae_count = basic_info.get('number_of_sae_subjects', 0)
                    st.metric("Serious AE Subjects", sae_count)
                
                with col3:
                    if nsae_count and sae_count:
                        total_ae = int(nsae_count) + int(sae_count)
                        st.metric("Total AE Subjects", total_ae)
                
                # Detailed adverse events (if available)
                if not report_data['adverse_events'].empty:
                    st.subheader("📊 Detailed Adverse Events")
                    
                    ae_df = report_data['adverse_events'].copy()
                    
                    # Show summary table
                    if len(ae_df) > 0:
                        st.dataframe(ae_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("Detailed adverse event data not available.")
                else:
                    st.info("Detailed adverse event information not available in current database.")
            
            # Tab 7: Results & Publications
            with tabs[6]:
                st.subheader("📊 Study Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    results_reported = 'Yes' if basic_info.get('were_results_reported') else 'No'
                    st.metric("Results Reported", results_reported)
                
                with col2:
                    months_to_report = basic_info.get('months_to_report_results', 'N/A')
                    st.metric("Months to Report", months_to_report)
                
                with col3:
                    fda_regulated = 'Yes' if basic_info.get('is_fda_regulated_drug') or basic_info.get('is_fda_regulated_device') else 'No'
                    st.metric("FDA Regulated", fda_regulated)
                
                # Outcome measurements (if available)
                if not report_data['outcome_measurements'].empty:
                    st.subheader("📈 Outcome Measurements")
                    outcome_measures = report_data['outcome_measurements']
                    st.dataframe(outcome_measures, use_container_width=True, hide_index=True)
                else:
                    st.info("Detailed outcome measurement data not available.")
                
                # Publications and references
                if not report_data['references'].empty:
                    st.subheader("📚 Publications & References")
                    references = report_data['references']
                    st.dataframe(references, use_container_width=True, hide_index=True)
                else:
                    st.info("No publication references available.")
                
                # Baseline measurements
                if not report_data['baseline_measurements'].empty:
                    st.subheader("📊 Baseline Characteristics")
                    baseline = report_data['baseline_measurements']
                    st.dataframe(baseline, use_container_width=True, hide_index=True)
            
            # Tab 8: Visualizations
            with tabs[7]:
                if include_visualizations:
                    st.subheader("📈 Study Visualizations")
                    
                    with st.spinner("Generating visualizations..."):
                        figures = create_study_report_visualizations(
                            report_data, selected_nct_id
                        )
                    
                    if figures:
                        for title, fig in figures:
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No visualizations available for this study.")
                else:
                    st.info("Visualizations disabled. Enable in report settings to view charts.")
            
            # Export functionality
            st.markdown("---")
            st.subheader("📥 Export Report")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📄 Download Report Summary"):
                    summary_text = generate_study_report_summary(
                        report_data, selected_nct_id
                    )
                    st.download_button(
                        label="⬇️ Download Summary",
                        data=summary_text,
                        file_name=f"clinical_study_report_{selected_nct_id}_{datetime.now().strftime('%Y%m%d')}.md",
                        mime="text/markdown"
                    )
            
            with col2:
                if st.button("📊 Download Study Data"):
                    # Create comprehensive data export
                    export_data = {
                        'basic_info': report_data['basic_info'].to_dict('records'),
                        'sponsors': report_data['sponsors'].to_dict('records'),
                        'facilities': report_data['facilities'].to_dict('records'),
                        'conditions': report_data['conditions'].to_dict('records'),
                        'interventions': report_data['interventions'].to_dict('records'),
                        'outcomes': report_data['outcomes'].to_dict('records'),
                        'investigators': report_data['investigators'].to_dict('records')
                    }
                    
                    import json
                    json_data = json.dumps(export_data, indent=2, default=str)
                    # csv_data=pd.DataFrame(export_data).to_csv(f"study_data_{selected_nct_id}_{datetime.now().strftime('%Y%m%d')}.csv",index=False)
                    
                    st.download_button(
                        label="⬇️ Download JSON",
                        data=json_data,
                        file_name=f"study_data_{selected_nct_id}_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )
                    # st.download_button(
                    #     label="⬇️ Download CSV",
                    #     data=csv_data,
                    #     # file_name=f"study_data_{selected_nct_id}_{datetime.now().strftime('%Y%m%d')}.csv",
                    #     mime="text/csv"
                    # )
            
            with col3:
                if st.button("📋 Copy Study ID"):
                    st.code(selected_nct_id)
                    st.success(f"Study ID {selected_nct_id} ready to copy!")
            
        else:
            st.error("Failed to generate study report. Please try a different study.")
    
    elif not st.session_state.get('generate_report', False):
                st.info("👆 Click 'Generate Comprehensive Report' to create detailed study analysis.")


# @st.fragment
def study_selection_widget(selected_sponsor_name):
    if selected_sponsor_name:
        # Load studies for selected sponsor
        studies_df = get_studies_by_sponsor(selected_sponsor_name)       
        if not studies_df.empty:
            # Create display format for studies
            study_options = []
            for _, study in studies_df.iterrows():
                study_display = f"{study['nct_id']}-[{study['overall_status']}] {study['brief_title'][:50]}{'...' if len(study['brief_title']) > 50 else ''}"
                study_options.append(study_display)
            
            selected_study_display = st.selectbox(
                "Select Study:",
                options=study_options,key="study_selection",
                help="Choose a specific study for detailed report"
            )
            
            # Extract NCT ID
            selected_nct_id = selected_study_display.split('-')[0]
            
            # Display study preview
            study_info = studies_df[studies_df['nct_id'] == selected_nct_id].iloc[0]
            st.markdown(f"""
                        **NCT ID:** {study_info['nct_id']}  
                        **Status:** {study_info['overall_status']}  
                        **Phase:** {study_info['phase']}  
                        **Enrollment:** {study_info['enrollment']}  
                        **Start Date:** {study_info['start_date']}
                        """)
            
            return selected_nct_id, study_info
        else:
            st.warning(f"No studies found for sponsor: {selected_sponsor_name}")
            return None, None
    else:
        return None, None
 # @st.fragment
def sponsor_selection_widget():
    sponsors_df =get_sponsors_list()
    if not sponsors_df.empty:
      
        selected_sponsor_display = st.selectbox(
            "Select Sponsor:",
            options=sponsors_df['sponsor_display'],index=None, key="sponsor_selection",
            help="Choose a sponsor to view their studies"
        )
        if not selected_sponsor_display:    
            return None, None
        else:
            # Extract actual sponsor name
            selected_sponsor_name = selected_sponsor_display.split(' [')[0]
        
            # Display sponsor info
            sponsor_info = sponsors_df[sponsors_df['sponsor_name'] == selected_sponsor_name].iloc[0]
            
            st.markdown(f"""
            **Selected Sponsor:** {sponsor_info['sponsor_name']}  
            **Agency Class:** {sponsor_info['agency_class']}  
            **Total Studies:** {sponsor_info['total_studies']}  
            **Completed Studies:** {sponsor_info['completed_studies']}
            """)
            return selected_sponsor_name, sponsor_info
    else:
        st.error("No sponsors found in the database.")
        return None, None

def render_study_reports_page():
    """Render the study reports page"""
    
    st.header("📄 Clinical Study Reports",divider=True)
    
     # Check if database is connected
    if not check_database_connection():
        return
        
    st.markdown("""
    <div class="insight-box">
    <h5>📊 Comprehensive Study Reporting</h5>
    <p>Generate detailed clinical study reports with comprehensive information including study design, 
    sites, investigators, outcomes, adverse events, and much more from the AACT database.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 1: Sponsor Selection
    st.subheader("🔍 Study Selection")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        selected_sponsor_name, sponsor_info = sponsor_selection_widget()
    
    with col2:
        selected_nct_id, study_info = study_selection_widget(selected_sponsor_name)
        

    # Call the function in place of the original code
    if selected_nct_id:
        comprehensive_report_section(selected_nct_id)
    else:
        st.info("👆 Please select a sponsor and study to generate a comprehensive clinical study report.")
render_study_reports_page()