#!/usr/bin/env python3
"""
Clinical Trial Location Optimizer - Streamlit Web Application

This interactive web application identifies optimal locations for future clinical trials 
by correlating historical trial data from the AACT database with regional population demographics.
This enhanced version provides detailed, context-aware help text for clinical research professionals.

Author: Clinical Research Data Analytics Team
Version: 2.3 (Clarified Population Metric)
Date: 2025-08-07
"""

import streamlit as st
import pandas as pd
import requests
import numpy as np
import logging
import re
import plotly.express as px
import io
import math
from datetime import datetime
import warnings
import pycountry

# Import custom modules
# Assuming these files exist in the specified paths
# from src.utils.common_service import check_database_connection
# from src.database.connection_manager import DatabaseConnectionManager

warnings.filterwarnings('ignore')

# --- Mock/Placeholder for custom modules if they are not available ---
# This allows the app to run without the original src directory.
# In a real environment, you would use the actual imports.

class DatabaseConnectionManager:
    """Mock DatabaseConnectionManager to allow the script to run without a live DB."""
    def execute_query(self, query, params=None):
        st.warning("Running with mock data. Please connect to a database for live analysis.")
        # Return an empty or sample DataFrame based on the query
        if "conditions" in query:
            return pd.DataFrame({'downcase_name': ['diabetes', 'cancer', 'hypertension', 'asthma']})
        if "countries" in query:
            return pd.DataFrame({'name': ['United States', 'Canada', 'Germany', 'Japan']})
        if "interventions" in query:
            return pd.DataFrame({'intervention_type': ['Drug', 'Device', 'Behavioral', 'Biological']})
        
        # For the main trial data query, return a sample DataFrame
        sample_data = {
            'condition': ['diabetes'] * 5,
            'facility_name': [f'Facility {i}' for i in range(5)],
            'facility_city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
            'facility_state': ['NY', 'CA', 'IL', 'TX', 'AZ'],
            'facility_country': ['United States'] * 5,
            'minimum_age': ['18 Years', '30 Years', '18 Years', '45 Years', '25 Years'],
            'maximum_age': ['65 Years', '75 Years', 'N/A', '70 Years', '60 Years'],
            'gender': ['All', 'Male', 'All', 'Female', 'All'],
            'healthy_volunteers': [False] * 5,
            'study_status': ['Completed'] * 5,
            'start_date': pd.to_datetime(['2020-01-01', '2019-05-15', '2021-02-20', '2018-11-10', '2022-03-01']),
            'completion_date': pd.to_datetime(['2022-01-01', '2021-05-15', '2023-02-20', '2020-11-10', '2024-03-01']),
            'intervention_type': ['Drug'] * 5,
            'lead_sponsor': [f'Sponsor {i}' for i in range(5)],
            'sponsor_type': ['Industry', 'NIH', 'Industry', 'Other', 'Industry'],
            'enrollment': [150, 200, 120, 250, 180],
            'enrollment_type': ['Actual'] * 5,
            'phase': ['Phase 3', 'Phase 2', 'Phase 3', 'Phase 4', 'Phase 2'],
            'nct_id': [f'NCT0000000{i}' for i in range(5)]
        }
        return pd.DataFrame(sample_data)

    def disconnect(self):
        pass

def check_database_connection():
    """Mock check_database_connection."""
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseConnectionManager()
        st.session_state.database_loaded = True
    return st.session_state.database_loaded

# --- End of Mock/Placeholder section ---


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()] # Log to console for Streamlit Cloud
)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="Clinical Trial Location Optimizer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PopulationDataFetcher:
    """Fetches and processes population data from public APIs."""
    
    def __init__(self):
        self.world_bank_base_url = "https://api.worldbank.org/v2"
        self.census_base_url = "https://api.census.gov/data"
    
    def fetch_world_bank_data(self, country_code: str, year: int = 2022) -> dict:
        """Fetches country-level population data from the World Bank API."""
        try:
            indicators = {
                'SP.POP.0014.MA.IN': 'male_0_14', 'SP.POP.1564.MA.IN': 'male_15_64', 
                'SP.POP.65UP.MA.IN': 'male_65_plus', 'SP.POP.0014.FE.IN': 'female_0_14',
                'SP.POP.1564.FE.IN': 'female_15_64', 'SP.POP.65UP.FE.IN': 'female_65_plus'
            }
            population_data = {}
            for indicator, key in indicators.items():
                url = f"{self.world_bank_base_url}/country/{country_code}/indicator/{indicator}"
                params = {'format': 'json', 'date': str(year)}
                response = requests.get(url, params=params, timeout=60)
                response.raise_for_status()
                data = response.json()
                if len(data) > 1 and data[1] and data[1][0]['value']:
                    population_data[key] = int(data[1][0]['value'])
                else:
                    population_data[key] = 0
            logger.info(f"Successfully fetched World Bank data for {country_code}")
            return population_data
        except Exception as e:
            logger.error(f"Failed to fetch World Bank data for {country_code}: {e}")
            return {}
    
    def fetch_us_census_data(self, state_code: str, year: int = 2022) -> dict:
        """Fetches US state-level population data from the Census Bureau API."""
        try:
            url = f"{self.census_base_url}/{year}/acs/acs5"
            variables = {
                'B01001_003E': 'male_under_5', 'B01001_004E': 'male_5_9', 'B01001_005E': 'male_10_14',
                'B01001_006E': 'male_15_17', 'B01001_007E': 'male_18_19', 'B01001_008E': 'male_20',
                'B01001_009E': 'male_21', 'B01001_010E': 'male_22_24', 'B01001_011E': 'male_25_29',
                'B01001_012E': 'male_30_34', 'B01001_013E': 'male_35_39', 'B01001_014E': 'male_40_44',
                'B01001_015E': 'male_45_49', 'B01001_016E': 'male_50_54', 'B01001_017E': 'male_55_59',
                'B01001_018E': 'male_60_61', 'B01001_019E': 'male_62_64', 'B01001_020E': 'male_65_66',
                'B01001_021E': 'male_67_69', 'B01001_022E': 'male_70_74', 'B01001_023E': 'male_75_79',
                'B01001_024E': 'male_80_84', 'B01001_025E': 'male_85_plus', 'B01001_027E': 'female_under_5',
                'B01001_028E': 'female_5_9', 'B01001_029E': 'female_10_14', 'B01001_030E': 'female_15_17',
                'B01001_031E': 'female_18_19', 'B01001_032E': 'female_20', 'B01001_033E': 'female_21',
                'B01001_034E': 'female_22_24', 'B01001_035E': 'female_25_29', 'B01001_036E': 'female_30_34',
                'B01001_037E': 'female_35_39', 'B01001_038E': 'female_40_44', 'B01001_039E': 'female_45_49',
                'B01001_040E': 'female_50_54', 'B01001_041E': 'female_55_59', 'B01001_042E': 'female_60_61',
                'B01001_043E': 'female_62_64', 'B01001_044E': 'female_65_66', 'B01001_045E': 'female_67_69',
                'B01001_046E': 'female_70_74', 'B01001_047E': 'female_75_79', 'B01001_048E': 'female_80_84',
                'B01001_049E': 'female_85_plus'
            }
            var_string = ','.join(variables.keys())
            params = {'get': var_string, 'for': f'state:{state_code}'}
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if len(data) > 1:
                result = dict(zip(data[0], data[1]))
                population_data = {var_name: int(result.get(var_code, '0')) for var_code, var_name in variables.items()}
                logger.info(f"Successfully fetched US Census data for state {state_code}")
                return population_data
            return {}
        except Exception as e:
            logger.error(f"Failed to fetch US Census data for state {state_code}: {e}")
            return {}

class AgeRangeProcessor:
    """Processes and standardizes age range formats and calculates population matches."""
    
    @staticmethod
    def parse_age_string(age_str: str) -> int | None:
        """Parses an age string (e.g., '18 Years') into a numeric value."""
        if not age_str or pd.isna(age_str) or str(age_str).upper() in ['N/A', 'NA', 'NULL']:
            return None
        numbers = re.findall(r'\d+', str(age_str))
        return int(numbers[0]) if numbers else None
    
    @staticmethod
    def standardize_age_range(min_age: str, max_age: str) -> tuple[int, int]:
        """Converts min/max age strings to a standardized numeric tuple."""
        min_val = AgeRangeProcessor.parse_age_string(min_age) or 18
        max_val = AgeRangeProcessor.parse_age_string(max_age) or 100
        return min_val, max_val
    
    @staticmethod
    def calculate_matching_population(population_data: dict, min_age: int, max_age: int, gender: str) -> int:
        """Calculates the matching population based on age, gender, and available demographic data."""
        total_population = 0
        us_census_age_groups = {
            'male': [('male_18_19', 18, 19), ('male_20', 20, 20), ('male_21', 21, 21), ('male_22_24', 22, 24), ('male_25_29', 25, 29), ('male_30_34', 30, 34), ('male_35_39', 35, 39), ('male_40_44', 40, 44), ('male_45_49', 45, 49), ('male_50_54', 50, 54), ('male_55_59', 55, 59), ('male_60_61', 60, 61), ('male_62_64', 62, 64), ('male_65_66', 65, 66), ('male_67_69', 67, 69), ('male_70_74', 70, 74), ('male_75_79', 75, 79), ('male_80_84', 80, 84), ('male_85_plus', 85, 100)],
            'female': [('female_18_19', 18, 19), ('female_20', 20, 20), ('female_21', 21, 21), ('female_22_24', 22, 24), ('female_25_29', 25, 29), ('female_30_34', 30, 34), ('female_35_39', 35, 39), ('female_40_44', 40, 44), ('female_45_49', 45, 49), ('female_50_54', 50, 54), ('female_55_59', 55, 59), ('female_60_61', 60, 61), ('female_62_64', 62, 64), ('female_65_66', 65, 66), ('female_67_69', 67, 69), ('female_70_74', 70, 74), ('female_75_79', 75, 79), ('female_80_84', 80, 84), ('female_85_plus', 85, 100)]
        }
        world_bank_age_groups = {
            'male': [('male_15_64', 15, 64), ('male_65_plus', 65, 100)],
            'female': [('female_15_64', 15, 64), ('female_65_plus', 65, 100)]
        }
        
        age_groups = us_census_age_groups if any(key.startswith('male_18_19') for key in population_data.keys()) else world_bank_age_groups
        
        target_genders = ['male', 'female'] if gender.upper() in ['ALL', 'BOTH'] else [gender.lower()]

        for target_gender in target_genders:
            if target_gender in age_groups:
                for group_key, group_min, group_max in age_groups[target_gender]:
                    if group_max >= min_age and group_min <= max_age:
                        population = population_data.get(group_key, 0)
                        overlap_min = max(group_min, min_age)
                        overlap_max = min(group_max, max_age)
                        overlap_years = max(0, overlap_max - overlap_min + 1)
                        group_years = group_max - group_min + 1
                        overlap_ratio = overlap_years / group_years if group_years > 0 else 0
                        total_population += int(population * overlap_ratio)
        return total_population

class ClinicalTrialOptimizer:
    """Main class for the clinical trial location optimization analysis."""
    
    def __init__(self):
        self.db_connector = st.session_state.db_manager
        self.population_fetcher = PopulationDataFetcher()
        self.age_processor = AgeRangeProcessor()

    def get_trial_data(self, condition_name: str, target_country: str = None, intervention_type: str = None) -> pd.DataFrame:
        """Fetches clinical trial data from the AACT database for a specific condition."""
        target_country_condition = "AND fac.country=:target_country" if target_country else ""
        intervention_type_condition = "AND inv.intervention_type=:intervention_type" if intervention_type else ""
        query = f"""
        SELECT DISTINCT
            c.name AS condition, fac.name AS facility_name, fac.city AS facility_city,
            fac.state AS facility_state, fac.country AS facility_country,
            e.minimum_age, e.maximum_age, e.gender, e.healthy_volunteers,
			s.overall_status AS study_status, s.start_date, s.completion_date,
			inv.intervention_type, sp.name as lead_sponsor, sp.agency_class as sponsor_type,
            s.enrollment, s.enrollment_type, s.phase, s.nct_id
        FROM conditions c
        JOIN studies s ON c.nct_id = s.nct_id
        JOIN eligibilities e ON s.nct_id = e.nct_id
        JOIN facilities fac ON s.nct_id = fac.nct_id
		JOIN interventions inv ON inv.nct_id=s.nct_id
		JOIN sponsors sp ON sp.nct_id=s.nct_id AND sp.lead_or_collaborator='lead'
        WHERE c.downcase_name LIKE LOWER(:condition)
            AND s.enrollment IS NOT NULL AND s.enrollment > 0
            AND fac.name IS NOT NULL AND fac.city IS NOT NULL AND fac.country IS NOT NULL
			{target_country_condition} {intervention_type_condition}
        ORDER BY s.enrollment DESC
        """
        params = {'condition': f"%{condition_name}%"}
        if target_country:
            params['target_country'] = target_country
        if intervention_type:
            params['intervention_type'] = intervention_type
        
        return self.db_connector.execute_query(query, params)
    
    def calculate_suitability_scores(self, trial_data: pd.DataFrame, target_country: str, 
                                     target_state: str = None, min_study_count: int = 2) -> pd.DataFrame:
        """Calculates suitability scores for potential trial locations."""
        results = []
        population_data = {}
        
        if target_country.upper() == 'UNITED STATES' and target_state:
            state_codes = {'alabama': '01', 'alaska': '02', 'arizona': '04', 'arkansas': '05', 'california': '06', 'colorado': '08', 'connecticut': '09', 'delaware': '10', 'florida': '12', 'georgia': '13', 'hawaii': '15', 'idaho': '16', 'illinois': '17', 'indiana': '18', 'iowa': '19', 'kansas': '20', 'kentucky': '21', 'louisiana': '22', 'maine': '23', 'maryland': '24', 'massachusetts': '25', 'michigan': '26', 'minnesota': '27', 'mississippi': '28', 'missouri': '29', 'montana': '30', 'nebraska': '31', 'nevada': '32', 'new hampshire': '33', 'new jersey': '34', 'new mexico': '35', 'new york': '36', 'north carolina': '37', 'north dakota': '38', 'ohio': '39', 'oklahoma': '40', 'oregon': '41', 'pennsylvania': '42', 'rhode island': '44', 'south carolina': '45', 'south dakota': '46', 'tennessee': '47', 'texas': '48', 'utah': '49', 'vermont': '50', 'virginia': '51', 'washington': '53', 'west virginia': '54', 'wisconsin': '55', 'wyoming': '56'}
            state_code = state_codes.get(target_state.lower())
            if state_code:
                population_data = self.population_fetcher.fetch_us_census_data(state_code)
        else:
            try:
                country = pycountry.countries.search_fuzzy(target_country)[0]
                population_data = self.population_fetcher.fetch_world_bank_data(country.alpha_2)
            except LookupError:
                logger.warning(f"Country code not found for {target_country}")
        
        if not population_data:
            st.error(f"Could not retrieve population data for {target_country}" + (f", {target_state}" if target_state else "") + ". Suitability scores cannot be calculated.")
            return pd.DataFrame()

        grouped_data = trial_data.groupby(['facility_name', 'facility_city', 'facility_state', 'facility_country', 'condition', 'minimum_age', 'maximum_age', 'gender']).agg(avg_enrollment=('enrollment', 'mean'), study_count=('nct_id', 'count')).reset_index()
        
        grouped_data = grouped_data[grouped_data['study_count'] >= min_study_count]
        
        for _, row in grouped_data.iterrows():
            min_age, max_age = self.age_processor.standardize_age_range(row['minimum_age'], row['maximum_age'])
            matching_population = self.age_processor.calculate_matching_population(population_data, min_age, max_age, row['gender'])
            
            if row['avg_enrollment'] > 0:
                raw_score = matching_population / row['avg_enrollment']
                suitability_score = min(100, (math.log10(raw_score + 1) / 5) * 100) if raw_score > 0 else 0
            else:
                suitability_score = 0
            
            results.append({
                'facility_name': row['facility_name'], 'facility_city': row['facility_city'],
                'facility_state': row['facility_state'], 'facility_country': row['facility_country'],
                'condition': row['condition'], 'required_age_range': f"{min_age}-{max_age} years",
                'required_gender': row['gender'], 'avg_past_enrollment': round(row['avg_enrollment'], 1),
                'study_count': row['study_count'], 'estimated_matching_population': matching_population,
                'suitability_score': round(suitability_score, 2)
            })
        
        results_df = pd.DataFrame(results)
        return results_df.sort_values('suitability_score', ascending=False) if not results_df.empty else results_df

def create_visualizations(results_df: pd.DataFrame):
    """Creates interactive Plotly visualizations for the analysis results."""
    if results_df.empty: return

    st.subheader("Top 10 Facilities by Suitability Score")
    st.info("""
    **What This Shows:** This chart ranks the top 10 facilities based on their calculated **Suitability Score**, which indicates the highest potential for successful patient recruitment.
    
    **How to Read:**
    - **Y-Axis:** The name of the clinical trial facility.
    - **X-Axis:** The Suitability Score. A higher score signifies a better ratio of available population to historical enrollment needs.
    - **Color:** Darker bars indicate higher scores, highlighting the most promising locations at a glance.
    
    **Data Insight:** Use this chart to quickly identify top-performing sites that have a strong track record and are located in areas with a large potential patient pool.
    """)
    top_10 = results_df.head(10)
    fig_bar = px.bar(top_10, x='suitability_score', y='facility_name', title='Top 10 Facilities by Suitability Score', labels={'suitability_score': 'Suitability Score', 'facility_name': 'Facility'}, color='suitability_score', color_continuous_scale='viridis', orientation='h')
    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Population vs. Past Enrollment Analysis")
    st.info("""
    **What This Shows:** This scatter plot visualizes the relationship between the **estimated available patient population** and the **average historical enrollment** for each facility. It helps identify sites that may be under-utilized.
    
    **How to Read:**
    - **X-Axis:** Average number of participants enrolled in past studies at the facility.
    - **Y-Axis:** Estimated number of people in the region matching the trial's demographic criteria (age, gender).
    - **Bubble Size & Color:** Represents the Suitability Score. Larger, darker bubbles signify higher recruitment potential.
    
    **Data Insight:** Look for facilities in the **upper-left quadrant** (high population, low past enrollment). These sites represent untapped potential, as they are in regions with many potential participants but have not historically enrolled large numbers.
    """)
    fig_scatter = px.scatter(results_df, x='avg_past_enrollment', y='estimated_matching_population', size='suitability_score', color='suitability_score', hover_data=['facility_name', 'facility_city', 'required_age_range'], title='Population vs. Past Enrollment', labels={'avg_past_enrollment': 'Average Past Enrollment (from AACT)', 'estimated_matching_population': 'Estimated Matching Population (from API)'}, color_continuous_scale='viridis')
    st.plotly_chart(fig_scatter, use_container_width=True)

def display_summary_metrics(results_df: pd.DataFrame):
    """Displays key summary metrics with detailed, context-rich help information."""
    if results_df.empty: return
    
    st.subheader("Analysis Summary Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Locations Analyzed",
            value=len(results_df),
            help="""
            **What it is:** The total number of unique facilities that met the initial analysis criteria based on your selections.
            
            **How it's Calculated:** This is a count of distinct facilities from the AACT database that have conducted at least the 'Minimum Study Count' for the specified medical condition, after grouping by demographic criteria.
            
            **Data Source:** AACT Database (facilities, studies, conditions tables).
            """
        )
    
    with col2:
        st.metric(
            label="Highest Suitability Score", 
            value=f"{results_df['suitability_score'].max():.1f}" if not results_df.empty else "N/A",
            help="""
            **What it is:** The top score achieved by a facility, indicating the best-identified recruitment potential.
            
            **Formula:** `Suitability Score = log10( (Estimated Matching Population / Average Past Enrollment) + 1 )` normalized to a 0-100 scale.
            
            **Interpretation:**
            - **> 50:** Excellent potential (large population, efficient enrollment).
            - **25-50:** Good potential.
            - **< 25:** Moderate to challenging potential.
            
            **Data Sources:** AACT (enrollment) & Population APIs (Census/World Bank).
            """
        )
    
    with col3:
        st.metric(
            label="Average Suitability Score",
            value=f"{results_df['suitability_score'].mean():.1f}" if not results_df.empty else "N/A",
            help="""
            **What it is:** The mean of all suitability scores for the analyzed locations.
            
            **Purpose:** This metric provides a high-level indicator of the overall quality of the recruitment landscape for your search criteria.
            
            **Interpretation:**
            - A **high average** suggests a favorable environment with many strong candidate sites.
            - A **low average** may indicate a competitive or saturated market, or a limited patient pool.
            """
        )
    
    with col4:
        total_population = results_df['estimated_matching_population'].sum()
        st.metric(
            label="Total Estimated Patient Pool",
            value=f"{total_population:,.0f}",
            help="""
            **What it is:** The aggregate estimated patient pool across all analyzed locations, based on their specific demographic requirements.
            
            **How it's Calculated:** This is the sum of the 'Estimated Matching Population' for each facility. The estimation process involves:
            1. Extracting age/gender criteria from historical trials in AACT.
            2. Querying the appropriate demographic API (US Census for US states, World Bank for countries).
            3. Applying a proportional matching algorithm to align broad API age groups with specific trial criteria.
            
            **Note:** This is an *estimate* of the potential pool, not a guarantee of enrollment. It is an aggregation of individual site estimates, not the total population of the country.
            """
        )

def download_csv(results_df: pd.DataFrame, condition_name: str, target_country: str):
    """Creates a download button for the results DataFrame as a CSV file."""
    if results_df.empty: return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"trial_feasibility_{condition_name.replace(' ', '_')}_{target_country.replace(' ', '_')}_{timestamp}.csv"
    
    csv_buffer = io.StringIO()
    results_df.to_csv(csv_buffer, index=False)
    
    st.download_button(
        label="📥 Download Full Results as CSV",
        data=csv_buffer.getvalue(),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
        help="""
        **What You Get:** A comprehensive CSV file containing all the data from the 'Detailed Results' table.
        
        **File Contents:**
        - All facility and location details.
        - The specific demographic criteria (age, gender) for each analysis group.
        - Historical performance metrics (avg enrollment, study count).
        - Population estimates and the final calculated Suitability Score.
        
        **Use Cases:**
        - In-depth analysis in Excel, R, or Python.
        - Importing into other business intelligence tools.
        - Archiving results for documentation and reporting.
        """
    )

def analysis_params():
    """Renders the input parameter widgets for the analysis."""
    col1, col2, col3, col4 = st.columns([3, 2, 2, 1], vertical_alignment='bottom')
    
    with col1:
        condition_name = st.selectbox(
            "Medical Condition",
            options=get_condition_names(),
            index=None,
            placeholder="e.g., Diabetes, Alzheimer's Disease",
            help="""
            **What it is:** The therapeutic area for the analysis. The list is populated with conditions from the AACT database.
            
            **How it works:** Your selection filters for historical studies related to this condition. The search uses a 'LIKE' query, so broader terms (e.g., 'Cancer') will yield more results than specific ones (e.g., 'Non-small cell lung cancer').
            
            **Data Source:** `conditions.downcase_name` from the AACT database.
            """
        )
    with col2:
        target_country = st.selectbox(
            "Target Country",
            options=get_distinct_countries(),
            index=None,
            placeholder="e.g., United States",
            help="""
            **What it is:** The primary geographic region for the feasibility analysis.
            
            **Impact on Data Source:** This selection determines which population API is used:
            - **United States:** Triggers state-level data requests to the **US Census Bureau API (ACS)**.
            - **Other Countries:** Triggers country-level data requests to the **World Bank API**.
            
            **Effect on Results:** The Suitability Score is heavily dependent on the population data from the selected country.
            """
        )
    with col3:
        intervention_type = st.selectbox(
            "Intervention Type (Optional)",
            options=[None] + get_distinct_inv_types(),
            index=0,
            format_func=lambda x: 'All Types' if x is None else x,
            help="""
            **What it is:** The type of intervention (e.g., Drug, Device) used in the historical trials.
            
            **How it works:** This acts as an additional filter on the AACT study data. Selecting a specific type can refine the analysis to be more comparable to your planned trial.
            
            **Data Source:** `interventions.intervention_type` from the AACT database. Leave as 'All Types' for a broader analysis.
            """
        )
    
    with st.expander("⚙️ Advanced Analysis Settings"):
        adv_col1, adv_col2 = st.columns(2)
        with adv_col1:
            min_study_count = st.number_input(
                "Minimum Historical Studies per Site",
                min_value=1, max_value=10, value=2,
                help="""
                **What it is:** A data quality filter to ensure statistical reliability.
                
                **How it works:** It excludes facilities that have conducted fewer than this number of studies for a given demographic profile. This prevents one-off, outlier enrollment numbers from skewing the analysis.
                
                **Recommendation:**
                - **2 (Default):** A good balance between data reliability and the number of included sites.
                - **3-5:** For higher confidence in the 'Average Past Enrollment' metric, at the cost of fewer results.
                """
            )
        with adv_col2:
            score_threshold = st.slider(
                "Minimum Suitability Score Threshold",
                min_value=0.0, max_value=100.0, value=0.0, step=5.0,
                help="""
                **What it is:** A filter to narrow down results to only the most promising locations.
                
                **How it works:** After all scores are calculated, the app will only display results at or above this threshold.
                
                **Recommendation:**
                - **0 (Default):** Show all potential locations.
                - **25+:** Focus on sites with good or better potential.
                - **50+:** Isolate only the top-tier, excellent potential sites.
                """
            )
    
    with col4:
        run_analysis_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    return condition_name, target_country, None, intervention_type, min_study_count, score_threshold, run_analysis_button

# Cached functions for fetching dropdown options
@st.cache_data(persist=True)
def get_condition_names():
    return st.session_state.db_manager.execute_query("SELECT DISTINCT downcase_name FROM conditions ORDER BY downcase_name")['downcase_name'].tolist()

@st.cache_data(persist=True)
def get_distinct_countries():
    return st.session_state.db_manager.execute_query("SELECT DISTINCT name FROM countries WHERE name IS NOT NULL ORDER BY name")['name'].tolist()

@st.cache_data(persist=True)
def get_distinct_inv_types():
    return st.session_state.db_manager.execute_query("SELECT DISTINCT intervention_type FROM interventions ORDER BY intervention_type")['intervention_type'].tolist()


def render_trail_feasibility_page():
    """Main function to render the Streamlit page."""
    
    st.title("🔬 Clinical Trial Feasibility & Site Selection")
    
    if not check_database_connection():
        st.error("Database connection failed. Please check your settings on the Home page.")
        return
        
    st.markdown("""
    This tool analyzes historical trial data from the AACT database and correlates it with regional population demographics 
    to identify the most suitable locations for future clinical trials.
    """)
 
    with st.container(border=True):
        st.subheader("📋 Analysis Parameters")
        condition_name, target_country, target_state, intervention_type, min_study_count, score_threshold, run_analysis = analysis_params()

    if 'results' not in st.session_state:
        st.session_state.results = pd.DataFrame()

    if run_analysis:
        if not condition_name or not target_country:
            st.error("❌ Please select a Medical Condition and a Target Country to run the analysis.")
            return
        
        optimizer = ClinicalTrialOptimizer()
        
        with st.spinner("Analyzing... Fetching trial data, calling population APIs, and calculating scores..."):
            try:
                trial_data = optimizer.get_trial_data(condition_name, target_country, intervention_type)
                
                if trial_data.empty:
                    st.warning(f"⚠️ No historical trial records found for **{condition_name}** in **{target_country}**. Try broadening your search criteria.")
                    st.session_state.results = pd.DataFrame()
                    return

                st.success(f"✅ Found {trial_data['nct_id'].nunique():,} unique studies across {len(trial_data):,} records.")
                
                with st.expander("View Raw Historical Trial Data"):
                    st.info("""
                    This table shows the raw data fetched from the AACT database for historical trials matching your criteria. Each row represents a specific facility's participation in a trial. This data forms the basis for the suitability analysis.
                    """)
                    st.markdown("""
                    | Column | Description | Data Source (AACT Table) |
                    |---|---|---|
                    | **nct_id** | Unique identifier for a clinical study. | `studies.nct_id` |
                    | **facility_name** | The name of the research institution or hospital. | `facilities.name` |
                    | **study_status** | The current recruitment status of the trial. | `studies.overall_status` |
                    | **enrollment** | The number of participants in the study. | `studies.enrollment` |
                    | **phase** | The phase of the clinical trial (e.g., Phase 1, Phase 2). | `studies.phase` |
                    | **lead_sponsor** | The primary organization responsible for the trial. | `sponsors.name` |
                    | **intervention_type** | The type of intervention being studied (e.g., Drug, Device). | `interventions.intervention_type` |
                    | **minimum_age / maximum_age**| The age eligibility criteria for participants. | `eligibilities` table |
                    """)
                    st.dataframe(trial_data)

                results = optimizer.calculate_suitability_scores(trial_data, target_country, target_state, min_study_count)
                
                if results.empty:
                    st.warning("⚠️ Analysis complete, but no locations met the specified criteria (e.g., minimum study count). Try adjusting the advanced settings.")
                    st.session_state.results = pd.DataFrame()
                    return

                if score_threshold > 0:
                    results = results[results['suitability_score'] >= score_threshold]
                
                st.session_state.results = results
                st.success("✅ Analysis complete!")

            except Exception as e:
                st.error(f"❌ An error occurred during analysis: {e}")
                logger.error(f"Analysis failed: {e}", exc_info=True)
                st.session_state.results = pd.DataFrame()
    
    if not st.session_state.results.empty:
        results = st.session_state.results
        st.header("📈 Analysis Results", divider='rainbow')
        
        display_summary_metrics(results)
        create_visualizations(results)
        
        st.subheader("📋 Detailed Results")
        with st.expander("ℹ️ Column Definitions & Data Sources"):
            st.markdown("""
            | Column | Description | Data Source & Calculation |
            |---|---|---|
            | **Suitability Score** | A 0-100 score indicating recruitment potential. | **Calculated:** `log10(Population / Enrollment)` |
            | **Facility Name** | The name of the research institution or hospital. | **AACT:** `facilities.name` |
            | **Location** | The geographic location of the facility. | **AACT:** `facilities.city`, `state`, `country` |
            | **Required Age/Gender** | The demographic criteria from historical trials. | **AACT:** `eligibilities` table |
            | **Avg Past Enrollment** | The average number of participants per study at that site for the demographic group. | **AACT:** `studies.enrollment` (Mean) |
            | **Study Count** | The number of historical studies used to calculate the average enrollment. | **AACT:** `studies` table (Count) |
            | **Est. Matching Pop.** | The estimated number of people in the region matching the demographic criteria. | **API:** US Census or World Bank |
            """)
        
        st.dataframe(
            results.style.format({
                'suitability_score': '{:.1f}',
                'avg_past_enrollment': '{:.1f}',
                'estimated_matching_population': '{:,.0f}'
            }).background_gradient(subset=['suitability_score'], cmap='viridis'),
            use_container_width=True,
            height=500
        )
        
        download_csv(results, condition_name, target_country)


render_trail_feasibility_page()
