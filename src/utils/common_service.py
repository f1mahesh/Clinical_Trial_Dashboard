import streamlit as st
from urllib.request import urlopen
import certifi
import json
from urllib.parse import quote_plus
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


CACHE_TTL = 60 * 60 * 24  # 24 hours


@st.cache_data(ttl=CACHE_TTL,show_spinner="Loading sponsors list...")
def get_sponsors_list():
    query=st.session_state.config.get_query('common_service', 'sponsors_list')
    df=st.session_state.db_manager.execute_query(query)
    df['sponsor_display']=df['sponsor_name']+" ["+df['agency_class']+"] - "+df['total_studies'].astype(str)+" studies"
    return df

@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading studies by sponsor...")   
def get_studies_by_sponsor(sponsor_name):
    query = st.session_state.config.get_query('common_service', 'studies_by_sponsor')
    params = {'sponsor_name':sponsor_name}
    return st.session_state.db_manager.execute_query(query, params)

@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading trial landscape metrics...")
def get_trial_landscape_metrics():
    """Calculate comprehensive trial landscape metrics"""
    
    # Basic statistics
    total_studies = st.session_state.db_manager.execute_query(
        st.session_state.config.get_query('trial_landscape', 'total_studies')
    ).iloc[0]['count']
    
    # Phase distribution
    phase_dist = st.session_state.db_manager.execute_query(
        st.session_state.config.get_query('trial_landscape', 'phase_distribution')
    )
    
    # Status distribution
    status_dist = st.session_state.db_manager.execute_query(
        st.session_state.config.get_query('trial_landscape', 'status_distribution')
    )
    
    # Study type distribution
    type_dist = st.session_state.db_manager.execute_query(
        st.session_state.config.get_query('trial_landscape', 'study_type_distribution')
    )
    
    # Top conditions
    top_conditions = st.session_state.db_manager.execute_query(
        st.session_state.config.get_query('trial_landscape', 'top_conditions')
    )
    
    # Sponsor analysis
    sponsor_analysis = st.session_state.db_manager.execute_query(
        st.session_state.config.get_query('trial_landscape', 'sponsor_analysis')
    )
    
    # Geographic distribution
    geographic_dist = st.session_state.db_manager.execute_query(
        st.session_state.config.get_query('trial_landscape', 'geographic_distribution')
    )
    
    return {
        'total_studies': total_studies,
        'phase_distribution': phase_dist,
        'status_distribution': status_dist,
        'type_distribution': type_dist,
        'top_conditions': top_conditions,
        'sponsor_analysis': sponsor_analysis,
        'geographic_distribution': geographic_dist
    }


@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading success and completion metrics...")
def get_success_completion_metrics():
    """Calculate success and completion metrics"""
    
    # Completion rates by phase
    completion_by_phase = st.session_state.db_manager.execute_query(
        st.session_state.config.get_query('success_analytics', 'completion_by_phase')
    )
    
    # Results reporting compliance
    results_reporting = st.session_state.db_manager.execute_query(
        st.session_state.config.get_query('success_analytics', 'results_reporting')
    )
    
    # Duration analysis
    duration_analysis = st.session_state.db_manager.execute_query(
        st.session_state.config.get_query('success_analytics', 'duration_analysis')
    )
    
    return {
        'completion_by_phase': completion_by_phase,
        'results_reporting': results_reporting,
        'duration_analysis': duration_analysis
    }



@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading innovation metrics...")
def get_innovation_metrics():
        """Calculate innovation and market insight metrics"""
        
        # Intervention type trends
        intervention_trends = st.session_state.db_manager.execute_query(
            st.session_state.config.get_query('innovation_insights', 'intervention_trends')
        )
        
        # Novel vs established interventions (simplified classification)
        novel_interventions = st.session_state.db_manager.execute_query(
            st.session_state.config.get_query('innovation_insights', 'novel_interventions')
        )
        
        # Emerging therapeutic areas (conditions with growing interest)
        emerging_areas = st.session_state.db_manager.execute_query(
            st.session_state.config.get_query('innovation_insights', 'emerging_areas')
        )
    
        # Device vs drug focus
        device_drug_analysis = st.session_state.db_manager.execute_query(
            st.session_state.config.get_query('innovation_insights', 'device_drug_analysis')
        )
        
        
        return {
            'intervention_trends': intervention_trends,
            'novel_interventions': novel_interventions,
            'emerging_areas': emerging_areas,
            'device_drug_analysis': device_drug_analysis
        }


@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading access and equity metrics...")
def get_access_equity_metrics():
        """Calculate access and equity metrics"""
        
        # Geographic accessibility
        geographic_access = st.session_state.db_manager.execute_query(
            st.session_state.config.get_query('access_equity', 'geographic_access')
        )

        # Age group inclusion
        age_inclusion = st.session_state.db_manager.execute_query(
            st.session_state.config.get_query('access_equity', 'age_inclusion')
        )
        
        # Gender inclusion
        gender_inclusion = st.session_state.db_manager.execute_query(
            st.session_state.config.get_query('access_equity', 'gender_inclusion')
        )
        
        # Healthy volunteer inclusion
        healthy_volunteer_inclusion = st.session_state.db_manager.execute_query(
            st.session_state.config.get_query('access_equity', 'healthy_volunteer_inclusion')
        )
        
        
        return {
            'geographic_access': geographic_access,
            'age_inclusion': age_inclusion,
            'gender_inclusion': gender_inclusion,
            'healthy_volunteer_inclusion': healthy_volunteer_inclusion
        }

def check_database_connection():
    # Check if database is connected
    if not hasattr(st.session_state, 'database_loaded') or not st.session_state.database_loaded:
        st.warning("Please connect to a database from the sidebar to view this page.")
        return False
    return True


def create_visualizations(metrics, viz_type: str) :
    """Create Plotly visualizations based on metrics"""
    
    figures = []
    
    if viz_type == 'landscape':
        # Phase distribution pie chart
        if not metrics['phase_distribution'].empty:
            fig_phase = px.pie(
                metrics['phase_distribution'], 
                values='count', 
                names='phase',
                title='Clinical Trial Distribution by Phase'
            )
            fig_phase.update_layout(height=400)
            figures.append(('Phase Distribution', fig_phase))
        
        # Status distribution bar chart
        if not metrics['status_distribution'].empty:
            fig_status = px.bar(
                metrics['status_distribution'], 
                x='overall_status', 
                y='count',
                title='Clinical Trial Distribution by Status',
                color='count',
                color_continuous_scale='viridis'
            )
            fig_status.update_layout(height=450, xaxis_tickangle=-45)
            figures.append(('Status Distribution', fig_status))
        
        # Top conditions horizontal bar chart
        if not metrics['top_conditions'].empty:
            fig_conditions = px.bar(
                metrics['top_conditions'].head(10), 
                x='studies_count', 
                y='name',
                orientation='h',
                title='Top 10 Medical Conditions in Clinical Trials',
                color='studies_count',
                color_continuous_scale='plasma'
            )
            fig_conditions.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
            figures.append(('Top Conditions', fig_conditions))
        
        # Sponsor analysis
        if not metrics['sponsor_analysis'].empty:
            fig_sponsor = px.scatter(
                metrics['sponsor_analysis'],
                x='avg_enrollment',
                y='avg_duration',
                size='studies_count',
                color='source_class',
                title='Sponsor Analysis: Enrollment vs Duration',
                hover_data=['studies_count']
            )
            fig_sponsor.update_layout(height=400)
            figures.append(('Sponsor Analysis', fig_sponsor))
    
    elif viz_type == 'success':
        # Completion rates by phase
        if not metrics['completion_by_phase'].empty:
            fig_completion = px.bar(
                metrics['completion_by_phase'],
                x='phase',
                y='completion_rate',
                title='Completion Rates by Study Phase',
                color='completion_rate',
                color_continuous_scale='RdYlGn',
                text='completion_rate'
            )
            fig_completion.update_traces(texttemplate='%{text}%', textposition='outside')
            fig_completion.update_layout(height=450)
            figures.append(('Completion Rates', fig_completion))
        
        # Results reporting by sponsor type
        if not metrics['results_reporting'].empty:
            fig_reporting = px.bar(
                metrics['results_reporting'],
                x='source_class',
                y='reporting_rate',
                title='Results Reporting Rate by Sponsor Type',
                color='reporting_rate',
                color_continuous_scale='blues',
                text='reporting_rate'
            )
            fig_reporting.update_traces(texttemplate='%{text}%', textposition='outside')
            fig_reporting.update_layout(height=400)
            figures.append(('Results Reporting', fig_reporting))
        
        # Duration analysis
        if not metrics['duration_analysis'].empty:
            fig_duration = px.box(
                pd.melt(metrics['duration_analysis'], 
                        id_vars=['phase'], 
                        value_vars=['avg_duration_months']),
                x='phase',
                y='value',
                title='Study Duration Distribution by Phase'
            )
            fig_duration.update_layout(height=400, yaxis_title='Duration (Months)')
            figures.append(('Duration Analysis', fig_duration))
    
    elif viz_type == 'innovation':
        # Intervention type distribution
        if not metrics['intervention_trends'].empty:
            fig_intervention = px.pie(
                metrics['intervention_trends'],
                values='studies_count',
                names='intervention_type',
                title='Studies by Intervention Type'
            )
            fig_intervention.update_layout(height=400)
            figures.append(('Intervention Types', fig_intervention))
        
        # Emerging therapeutic areas
        if not metrics['emerging_areas'].empty:
            fig_emerging = px.bar(
                metrics['emerging_areas'].head(10),
                x='condition',
                y='studies_count',
                title='Emerging Therapeutic Areas (Top 10)',
                color='industry_studies',
                color_continuous_scale='viridis'
            )
            fig_emerging.update_layout(height=400, xaxis_tickangle=-45)
            figures.append(('Emerging Areas', fig_emerging))
    
    elif viz_type == 'access':
        # Geographic distribution
        if not metrics['geographic_access'].empty:
            fig_geo = px.bar(
                metrics['geographic_access'].head(10),
                x='country',
                y='studies_available',
                title='Clinical Trial Availability by Country',
                color='total_facilities',
                color_continuous_scale='viridis'
            )
            fig_geo.update_layout(height=400, xaxis_tickangle=-45)
            figures.append(('Geographic Access', fig_geo))
        
        # Gender inclusion
        if not metrics['gender_inclusion'].empty:
            fig_gender = px.pie(
                metrics['gender_inclusion'],
                values='studies_count',
                names='gender',
                title='Gender Inclusion in Clinical Trials'
            )
            fig_gender.update_layout(height=400)
            figures.append(('Gender Inclusion', fig_gender))
        
        # Age group inclusion
        if not metrics['age_inclusion'].empty:
            fig_age = px.bar(
                metrics['age_inclusion'],
                x='age_group',
                y='studies_count',
                title='Age Group Inclusion in Clinical Trials',
                color='studies_count',
                color_continuous_scale='plasma'
            )
            fig_age.update_layout(height=400, xaxis_tickangle=-45)
            figures.append(('Age Group Inclusion', fig_age))
    
    return figures
   

def get_jsonparsed_data(url:str):
    print("url",url)
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)

@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading stock data...")
def get_sponsors_stock_data(sponsor_name: str) -> pd.DataFrame:
    """Get stock data for a specific sponsor"""
    api_key="o8MFLjaKTv6npM6toFkzr9ekTfXpPg0P"
    
    encoded_sponsor_name = quote_plus(sponsor_name)
    search_url=f"https://financialmodelingprep.com/api/v3/search-name?query={encoded_sponsor_name}&apikey={api_key}"
    
    search_data=get_jsonparsed_data(search_url)
    stock_data=pd.json_normalize(search_data)
    if not stock_data.empty:
        # st.dataframe(stock_data,use_container_width=True,hide_index=True)
        stock_list = ['NASDAQ','NYSE']
        filtered_stocks = stock_data[stock_data['exchangeShortName'].isin(stock_list)]
        if not filtered_stocks.empty:
            symole = filtered_stocks.iloc[0]['symbol']
        else:
            return pd.DataFrame()
        profile_url=f"https://financialmodelingprep.com/api/v3/profile/{symole}?apikey={api_key}"
        profile_data=get_jsonparsed_data(profile_url)    
        profile_df=pd.json_normalize(profile_data)
        if not profile_df.empty:
            return profile_df
    return pd.DataFrame()