#!/usr/bin/env python3
"""
Clinical Trial Location Optimizer - Streamlit Web Application

This interactive web application identifies optimal locations for future clinical trials 
by correlating historical trial data with regional population demographics from the AACT database.

Author: Clinical Research Data Analytics Team
Version: 2.0 (Streamlit UI)
Date: 2025-01-30
"""

import streamlit as st
import psycopg2
import pandas as pd
import requests
import json
import numpy as np
import logging
import sys
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings
import io
import math
warnings.filterwarnings('ignore')
from src.utils.common_service import check_database_connection
# Import database connection manager
from src.database.connection_manager import DatabaseConnectionManager
import pycountry
from millify import millify

# Configure logging for Streamlit
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Logs/trail_feasibility.log')
    ]
)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="Clinical Trial Location Optimizer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class TrialDemographics:
    """Data class for storing trial demographic requirements"""
    condition: str
    facility_name: str
    facility_city: str
    facility_state: str
    facility_country: str
    min_age: str
    max_age: str
    gender: str
    healthy_volunteers: bool
    study_count: int
    avg_enrollment: float

@dataclass
class PopulationData:
    """Data class for storing population demographic data"""
    country: str
    state: str
    age_group: str
    gender: str
    population: int



class PopulationDataFetcher:
    """Fetches population data from various APIs"""
    
    def __init__(self):
        self.world_bank_base_url = "https://api.worldbank.org/v2"
        self.census_base_url = "https://api.census.gov/data"
    
    def fetch_world_bank_data(self, country_code: str, year: int = 2022) -> Dict:
        """
        Fetch population data from World Bank API
        
        Args:
            country_code: ISO 3166-1 alpha-2 country code
            year: Year for data retrieval
            
        Returns:
            Dict: Population data by age and gender
        """
        try:
            # World Bank population by age groups
            indicators = {
                'SP.POP.0014.MA.IN': 'male_0_14',
                'SP.POP.1564.MA.IN': 'male_15_64', 
                'SP.POP.65UP.MA.IN': 'male_65_plus',
                'SP.POP.0014.FE.IN': 'female_0_14',
                'SP.POP.1564.FE.IN': 'female_15_64',
                'SP.POP.65UP.FE.IN': 'female_65_plus'
            }
            
            population_data = {}
            
            for indicator, key in indicators.items():
                url = f"{self.world_bank_base_url}/country/{country_code}/indicator/{indicator}"
                params = {'format': 'json', 'date': str(year)}
                logger.info(f"Fetching World Bank data for {country_code} with params: {params} and url: {url}")
                response = requests.get(url, params=params, timeout=60)
                response.raise_for_status()
                logger.info(f"Response: {response.json()}")
                data = response.json()
                if len(data) > 1 and data[1]:
                    value = data[1][0]['value'] if data[1][0]['value'] else 0
                    population_data[key] = int(value)
                else:
                    population_data[key] = 0
                    
            logger.info(f"Successfully fetched World Bank data for {country_code}")
            # st.write(population_data)
            return population_data
            
        except Exception as e:
            logger.error(f"Failed to fetch World Bank data: {e}")
            return {}
    
    def fetch_us_census_data(self, state_code: str, year: int = 2022) -> Dict:
        """
        Fetch US population data from Census Bureau API
        
        Args:
            state_code: US state FIPS code
            year: Year for data retrieval
            
        Returns:
            Dict: Population data by age and gender
        """
        try:
            # Using American Community Survey (ACS) 5-year estimates
            url = f"{self.census_base_url}/{year}/acs/acs5"
            
            # Age and sex variables from ACS
            variables = {
                'B01001_003E': 'male_under_5',
                'B01001_004E': 'male_5_9',
                'B01001_005E': 'male_10_14',
                'B01001_006E': 'male_15_17',
                'B01001_007E': 'male_18_19',
                'B01001_008E': 'male_20',
                'B01001_009E': 'male_21',
                'B01001_010E': 'male_22_24',
                'B01001_011E': 'male_25_29',
                'B01001_012E': 'male_30_34',
                'B01001_013E': 'male_35_39',
                'B01001_014E': 'male_40_44',
                'B01001_015E': 'male_45_49',
                'B01001_016E': 'male_50_54',
                'B01001_017E': 'male_55_59',
                'B01001_018E': 'male_60_61',
                'B01001_019E': 'male_62_64',
                'B01001_020E': 'male_65_66',
                'B01001_021E': 'male_67_69',
                'B01001_022E': 'male_70_74',
                'B01001_023E': 'male_75_79',
                'B01001_024E': 'male_80_84',
                'B01001_025E': 'male_85_plus',
                # Female categories
                'B01001_027E': 'female_under_5',
                'B01001_028E': 'female_5_9',
                'B01001_029E': 'female_10_14',
                'B01001_030E': 'female_15_17',
                'B01001_031E': 'female_18_19',
                'B01001_032E': 'female_20',
                'B01001_033E': 'female_21',
                'B01001_034E': 'female_22_24',
                'B01001_035E': 'female_25_29',
                'B01001_036E': 'female_30_34',
                'B01001_037E': 'female_35_39',
                'B01001_038E': 'female_40_44',
                'B01001_039E': 'female_45_49',
                'B01001_040E': 'female_50_54',
                'B01001_041E': 'female_55_59',
                'B01001_042E': 'female_60_61',
                'B01001_043E': 'female_62_64',
                'B01001_044E': 'female_65_66',
                'B01001_045E': 'female_67_69',
                'B01001_046E': 'female_70_74',
                'B01001_047E': 'female_75_79',
                'B01001_048E': 'female_80_84',
                'B01001_049E': 'female_85_plus'
            }
            
            var_string = ','.join(variables.keys())
            params = {
                'get': var_string,
                'for': f'state:{state_code}'
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if len(data) > 1:
                # Convert to dictionary
                headers = data[0]
                values = data[1]
                result = dict(zip(headers, values))
                
                # Map to our format
                population_data = {}
                for var_code, var_name in variables.items():
                    value = result.get(var_code, '0')
                    population_data[var_name] = int(value) if value != '-1' else 0
                
                logger.info(f"Successfully fetched US Census data for state {state_code}")
                return population_data
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to fetch US Census data: {e}")
            return {}

class AgeRangeProcessor:
    """Processes and standardizes age range formats"""
    
    @staticmethod
    def parse_age_string(age_str: str) -> Optional[int]:
        """
        Parse age string to extract numeric value
        
        Args:
            age_str: Age string (e.g., "18 Years", "65+", "N/A")
            
        Returns:
            Optional[int]: Numeric age value or None
        """
        if not age_str or pd.isna(age_str) or age_str.upper() in ['N/A', 'NA', 'NULL']:
            return None
        
        # Extract numbers from string
        numbers = re.findall(r'\d+', str(age_str))
        if numbers:
            return int(numbers[0])
        return None
    
    @staticmethod
    def standardize_age_range(min_age: str, max_age: str) -> Tuple[int, int]:
        """
        Standardize age range to numeric values
        
        Args:
            min_age: Minimum age string
            max_age: Maximum age string
            
        Returns:
            Tuple[int, int]: Standardized min and max ages
        """
        min_val = AgeRangeProcessor.parse_age_string(min_age)
        max_val = AgeRangeProcessor.parse_age_string(max_age)
        
        # Default values if parsing fails
        if min_val is None:
            min_val = 18  # Default minimum adult age
        if max_val is None:
            max_val = 100  # Default maximum age
            
        return min_val, max_val
    
    @staticmethod
    def calculate_matching_population(population_data: Dict, min_age: int, max_age: int, gender: str) -> int:
        """
        Calculate matching population for given criteria
        
        Args:
            population_data: Dictionary with population data
            min_age: Minimum age
            max_age: Maximum age
            gender: Target gender ('All', 'Male', 'Female')
            
        Returns:
            int: Matching population count
        """
        total_population = 0
        
        # Define age group mappings for different data sources
        us_census_age_groups = {
            'male': [
                ('male_18_19', 18, 19),
                ('male_20', 20, 20),
                ('male_21', 21, 21),
                ('male_22_24', 22, 24),
                ('male_25_29', 25, 29),
                ('male_30_34', 30, 34),
                ('male_35_39', 35, 39),
                ('male_40_44', 40, 44),
                ('male_45_49', 45, 49),
                ('male_50_54', 50, 54),
                ('male_55_59', 55, 59),
                ('male_60_61', 60, 61),
                ('male_62_64', 62, 64),
                ('male_65_66', 65, 66),
                ('male_67_69', 67, 69),
                ('male_70_74', 70, 74),
                ('male_75_79', 75, 79),
                ('male_80_84', 80, 84),
                ('male_85_plus', 85, 100)
            ],
            'female': [
                ('female_18_19', 18, 19),
                ('female_20', 20, 20),
                ('female_21', 21, 21),
                ('female_22_24', 22, 24),
                ('female_25_29', 25, 29),
                ('female_30_34', 30, 34),
                ('female_35_39', 35, 39),
                ('female_40_44', 40, 44),
                ('female_45_49', 45, 49),
                ('female_50_54', 50, 54),
                ('female_55_59', 55, 59),
                ('female_60_61', 60, 61),
                ('female_62_64', 62, 64),
                ('female_65_66', 65, 66),
                ('female_67_69', 67, 69),
                ('female_70_74', 70, 74),
                ('female_75_79', 75, 79),
                ('female_80_84', 80, 84),
                ('female_85_plus', 85, 100)
            ]
        }
        
        world_bank_age_groups = {
            'male': [
                ('male_15_64', 15, 64),
                ('male_65_plus', 65, 100)
            ],
            'female': [
                ('female_15_64', 15, 64),
                ('female_65_plus', 65, 100)
            ]
        }
        
        # Determine which age groups to use based on available data
        if any(key.startswith('male_18_19') for key in population_data.keys()):
            age_groups = us_census_age_groups
        else:
            age_groups = world_bank_age_groups
        
        # Calculate population based on gender criteria
        if gender.upper() in ['ALL', 'BOTH']:
            target_genders = ['male', 'female']
        elif gender.upper() in ['MALE', 'M']:
            target_genders = ['male']
        elif gender.upper() in ['FEMALE', 'F']:
            target_genders = ['female']
        else:
            target_genders = ['male', 'female']  # Default to both
        
        for target_gender in target_genders:
            for group_key, group_min, group_max in age_groups.get(target_gender, []):
                # Check if age group overlaps with target range
                if group_max >= min_age and group_min <= max_age:
                    population = population_data.get(group_key, 0)
                    
                    # Calculate overlap ratio for partial matches
                    overlap_min = max(group_min, min_age)
                    overlap_max = min(group_max, max_age)
                    overlap_years = max(0, overlap_max - overlap_min + 1)
                    group_years = group_max - group_min + 1
                    
                    overlap_ratio = overlap_years / group_years
                    total_population += int(population * overlap_ratio)
        
        return total_population
CACHE_TTL=60*60*24

class ClinicalTrialOptimizer:
    """Main class for clinical trial location optimization"""
    
    def __init__(self):
        self.db_connector = DatabaseConnectionManager()
        self.population_fetcher = PopulationDataFetcher()
        self.age_processor = AgeRangeProcessor()
    # @st.cache_data(ttl=CACHE_TTL, show_spinner="Loading trial data...")    
    def get_trial_data(self, condition_name: str,target_country: str = None,intervention_type: str = None) -> pd.DataFrame:
        """
        Fetch clinical trial data for specific condition
        
        Args:
            condition_name: Medical condition name
            target_country: Target country for analysis
        Returns:
            pd.DataFrame: Trial data with facility information
        """
        target_country_condition = f"AND fac.country=:target_country" if target_country else ""
        intervention_type_condition = f"AND inv.intervention_type=:intervention_type" if intervention_type else ""
        query = f"""
        SELECT DISTINCT
            c.name AS condition,
            fac.name AS facility_name,
            fac.city AS facility_city,
            fac.state AS facility_state,
            fac.country AS facility_country,
            e.minimum_age,
            e.maximum_age,
            e.gender,
            e.healthy_volunteers,
			s.overall_status AS study_status,
            s.start_date,
			s.completion_date,
			inv.intervention_type,
			sp.name as lead_sponsor,
			sp.agency_class as sponsor_type,
            s.enrollment,
			s.enrollment_type,
			s.phase,
            s.nct_id
        FROM
            conditions c
        JOIN
            studies s ON c.nct_id = s.nct_id
        JOIN
            eligibilities e ON s.nct_id = e.nct_id
        JOIN
            facilities fac ON s.nct_id = fac.nct_id
		join 
			interventions inv on inv.nct_id=s.nct_id
		join 
			sponsors sp on sp.nct_id=s.nct_id and sp.lead_or_collaborator='lead'
        WHERE 1=1
            --AND s.overall_status = 'COMPLETED' 
            AND c.downcase_name LIKE LOWER(:condition)
            AND s.enrollment IS NOT NULL
            AND s.enrollment > 0
            AND fac.name IS NOT NULL
            AND fac.city IS NOT NULL
            AND fac.country IS NOT NULL
			{target_country_condition}
			{intervention_type_condition}
        ORDER BY s.enrollment DESC
        """
        params = {
            'condition': f"%{condition_name}%",
            'target_country': target_country,
            'intervention_type': intervention_type,
        }
        
        return self.db_connector.execute_query(query, params)
    
    def calculate_suitability_scores(self, trial_data: pd.DataFrame, target_country: str, 
                                   target_state: str = None) -> pd.DataFrame:
        """
        Calculate suitability scores for trial locations
        
        Args:
            trial_data: DataFrame with trial information
            target_country: Target country for analysis
            target_state: Target state/province (optional)
            
        Returns:
            pd.DataFrame: Results with suitability scores
        """
        results = []
        logger.info(f"Target country: {target_country}")
        print("Target country:",target_country)
        
        # Get population data for target region
        if target_country.upper() == 'UNITED STATES' and target_state:
            # US Census data
            state_codes = {
                'alabama': '01', 'alaska': '02', 'arizona': '04', 'arkansas': '05',
                'california': '06', 'colorado': '08', 'connecticut': '09', 'delaware': '10',
                'florida': '12', 'georgia': '13', 'hawaii': '15', 'idaho': '16',
                'illinois': '17', 'indiana': '18', 'iowa': '19', 'kansas': '20',
                'kentucky': '21', 'louisiana': '22', 'maine': '23', 'maryland': '24',
                'massachusetts': '25', 'michigan': '26', 'minnesota': '27', 'mississippi': '28',
                'missouri': '29', 'montana': '30', 'nebraska': '31', 'nevada': '32',
                'new hampshire': '33', 'new jersey': '34', 'new mexico': '35', 'new york': '36',
                'north carolina': '37', 'north dakota': '38', 'ohio': '39', 'oklahoma': '40',
                'oregon': '41', 'pennsylvania': '42', 'rhode island': '44', 'south carolina': '45',
                'south dakota': '46', 'tennessee': '47', 'texas': '48', 'utah': '49',
                'vermont': '50', 'virginia': '51', 'washington': '53', 'west virginia': '54',
                'wisconsin': '55', 'wyoming': '56'
            }
            
            state_code = state_codes.get(target_state.lower())
            if state_code:
                population_data = self.population_fetcher.fetch_us_census_data(state_code)
            else:
                logger.warning(f"State code not found for {target_state}")
                population_data = {}
        else:
            # World Bank data (country level)
            
            
            try:
                country = pycountry.countries.search_fuzzy(target_country)[0] if target_country else None
                country_code = country.alpha_2 if country else None
            except LookupError:
                country_code = None
            logger.info(f"Country code: {country_code}")
            print(country_code)
            if country_code:
                population_data = self.population_fetcher.fetch_world_bank_data(country_code)
            else:
                logger.warning(f"Country code not found for {target_country}")
                population_data = {}
        
        if not population_data:
            logger.error("No population data available for the specified region")
            return pd.DataFrame()
        
        # Group by demographic criteria
        grouped_data = trial_data.groupby([
            'facility_name', 'facility_city', 'facility_state', 'facility_country',
            'condition', 'minimum_age', 'maximum_age', 'gender'
        ]).agg({
            'enrollment': ['mean', 'count'],
            'nct_id': 'count'
        }).reset_index()
        
        # Flatten column names
        grouped_data.columns = [
            'facility_name', 'facility_city', 'facility_state', 'facility_country',
            'condition', 'minimum_age', 'maximum_age', 'gender',
            'avg_enrollment', 'enrollment_count', 'study_count'
        ]
        
        # Filter for adequate sample size
        grouped_data = grouped_data[grouped_data['study_count'] >= 2]
        
        for _, row in grouped_data.iterrows():
            # Parse age range
            min_age, max_age = self.age_processor.standardize_age_range(
                row['minimum_age'], row['maximum_age']
            )
            
            # Calculate matching population
            matching_population = self.age_processor.calculate_matching_population(
                population_data, min_age, max_age, row['gender']
            )
            
            # Calculate normalized suitability score (0-100 scale)
            if row['avg_enrollment'] > 0:
                # Calculate raw score
                raw_score = matching_population / row['avg_enrollment']
                
                # Normalize using log scale to handle large population differences
                # Add 1 to avoid log(0), then normalize to 0-100 range
                suitability_score = min(100, (math.log10(raw_score + 1) / 5) * 100)
            else:
                suitability_score = 0
            
            # Create age range string
            age_range = f"{min_age}-{max_age} years"
            
            results.append({
                'facility_name': row['facility_name'],
                'facility_city': row['facility_city'],
                'facility_state': row['facility_state'],
                'facility_country': row['facility_country'],
                'condition': row['condition'],
                'required_age_range': age_range,
                'required_gender': row['gender'],
                'avg_past_enrollment': round(row['avg_enrollment'], 1),
                'study_count': row['study_count'],
                'estimated_matching_population': matching_population,
                'suitability_score': round(suitability_score, 2)
            })
        
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values('suitability_score', ascending=False)
        
        return results_df
    
    def run_analysis(self, condition_name: str, target_country: str, 
                    target_state: str = None, output_file: str = None) -> pd.DataFrame:
        """
        Run complete analysis workflow
        
        Args:
            condition_name: Medical condition to analyze
            target_country: Target country for analysis
            target_state: Target state/province (optional)
            output_file: Output CSV filename (optional)
            
        Returns:
            pd.DataFrame: Analysis results
        """
        logger.info(f"Starting analysis for condition: {condition_name}")
        logger.info(f"Target region: {target_country}" + (f", {target_state}" if target_state else ""))
        
        
        try:
            # Step 1: Get trial data
            logger.info("Fetching clinical trial data...")
            trial_data = self.get_trial_data(condition_name)
            
            if trial_data.empty:
                logger.warning(f"No completed trials found for condition: {condition_name}")
                return pd.DataFrame()
            
            logger.info(f"Found {len(trial_data)} trial records")
            
            # Step 2: Calculate suitability scores
            logger.info("Calculating suitability scores...")
            results = self.calculate_suitability_scores(trial_data, target_country, target_state)
            
            if results.empty:
                logger.warning("No results generated")
                return pd.DataFrame()
            
            # Step 3: Save results
            if output_file:
                results.to_csv(output_file, index=False)
                logger.info(f"Results saved to {output_file}")
            
            logger.info(f"Analysis completed. Found {len(results)} potential locations.")
            logger.info(f"Top location: {results.iloc[0]['facility_name']} (Score: {results.iloc[0]['suitability_score']})")
            
            return results
            
        finally:
            logger.info("Fetching clinical trial data finished...")

def create_visualizations(results_df: pd.DataFrame) -> None:
    """Create interactive visualizations for the results with detailed explanations"""
    
    if results_df.empty:
        return
    
    # Top 10 facilities by suitability score
    st.subheader("📊 Top 10 Facilities by Suitability Score")
    st.info("""
    **📈 What This Shows:** Facilities ranked by their recruitment potential
    
    **🔍 How to Read:** Higher scores (darker colors) indicate better recruitment prospects
    
    **📊 Data Source:** Suitability scores calculated from AACT enrollment data and regional demographics
    """)
    
    top_10 = results_df.head(10)
    
    # Horizontal bar chart of top facilities
    fig_bar = px.bar(
        top_10, 
        x='suitability_score', 
        y='facility_name',
        title='Top 10 Facilities by Suitability Score',
        labels={'suitability_score': 'Suitability Score (Population ÷ Past Enrollment)', 'facility_name': 'Facility'},
        color='suitability_score',
        color_continuous_scale='viridis'
    )
    fig_bar.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Scatter plot: Population vs Enrollment
    st.subheader("🔍 Population vs Past Enrollment Analysis")
    st.info("""
    **📈 What This Shows:** Relationship between available population and historical enrollment
    
    **🔍 How to Read:** 
    • **X-axis:** Average past enrollment per study (from AACT database)
    • **Y-axis:** Estimated matching population (from Census/World Bank APIs)
    • **Bubble Size:** Suitability score (larger = better recruitment potential)
    • **Color:** Suitability score (darker = higher score)
    
    **💡 Insight:** Facilities in upper-left (high population, low past enrollment) may offer untapped potential
    """)
    
    fig_scatter = px.scatter(
        results_df,
        x='avg_past_enrollment',
        y='estimated_matching_population',
        size='suitability_score',
        color='suitability_score',
        hover_data=['facility_name', 'facility_city', 'required_age_range'],
        title='Population vs Past Enrollment (Size = Suitability Score)',
        labels={
            'avg_past_enrollment': 'Average Past Enrollment (Historical Data)',
            'estimated_matching_population': 'Estimated Matching Population (Census/World Bank)'
        },
        color_continuous_scale='viridis'
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Distribution of suitability scores
    st.subheader("📈 Distribution of Suitability Scores")
    st.info("""
    **📈 What This Shows:** Overall quality distribution of potential recruitment locations
    
    **🔍 How to Read:**
    • **Right-skewed:** Many high-quality locations available
    • **Left-skewed:** Limited high-quality options
    • **Normal distribution:** Balanced recruitment landscape
    
    **📊 Score Interpretation:**
    • **>50:** Excellent potential | **25-50:** Good | **10-25:** Moderate | **<10:** Challenging
    """)
    
    fig_hist = px.histogram(
        results_df,
        x='suitability_score',
        nbins=20,
        title='Distribution of Suitability Scores',
        labels={'suitability_score': 'Suitability Score (Population ÷ Enrollment)', 'count': 'Number of Facilities'}
    )
    st.plotly_chart(fig_hist, use_container_width=True)

def display_summary_metrics(results_df: pd.DataFrame) -> None:
    """Display summary metrics in columns with detailed help information"""
    
    if results_df.empty:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Locations Found",
            value=len(results_df),
            help="""
            **How Calculated:** Count of unique facilities that meet analysis criteria
            
            **Data Source:** AACT database - facilities table joined with completed studies
            
            **Criteria Applied:**
            • Studies with 'COMPLETED' status
            • Facilities with complete location information
            • Minimum 2 historical studies per demographic group
            • Valid enrollment and demographic data
            """
        )
    
    with col2:
        st.metric(
            label="Highest Suitability Score", 
            value=f"{results_df['suitability_score'].max():.1f}",
            help="""
            **How Calculated:** Maximum suitability score among all facilities
            
            **Suitability Score Formula:**
            Estimated Matching Population ÷ Average Past Enrollment
            
            **Data Sources:**
            • Population: US Census Bureau API or World Bank API
            • Enrollment: AACT database historical averages
            
            **Interpretation:**
            • Score > 50: Excellent recruitment potential
            • Score 25-50: Good potential
            • Score 10-25: Moderate potential
            • Score < 10: Challenging recruitment
            """
        )
    
    with col3:
        st.metric(
            label="Average Score",
            value=f"{results_df['suitability_score'].mean():.1f}",
            help="""
            **How Calculated:** Mean of all facility suitability scores
            
            **Purpose:** Indicates overall recruitment landscape quality
            
            **Data Source:** Calculated from individual facility scores
            
            **Context:**
            • Higher average suggests favorable recruitment environment
            • Lower average may indicate market saturation or limited population
            • Compare across regions for strategic planning
            """
        )
    
    # with col4:
    #     total_population = results_df['estimated_matching_population'].sum()
    #     st.metric(
    #         label="Total Estimated Patient Pool",
    #         value=f"{total_population:,.0f} ({millify(total_population)})",
    #         help="""
    #         **What it is:** The aggregate estimated patient pool across all analyzed locations, based on their specific demographic requirements.
            
    #         **How it's Calculated:** This is the sum of the 'Estimated Matching Population' for each facility. The estimation process involves:
    #         1. Extracting age/gender criteria from historical trials in AACT.
    #         2. Querying the appropriate demographic API (US Census for US states, World Bank for countries).
    #         3. Applying a proportional matching algorithm to align broad API age groups with specific trial criteria.
            
    #         **Note:** This is an *estimate* of the potential pool, not a guarantee of enrollment. It is an aggregation of individual site estimates, not the total population of the country.
    #         """
    #     )
        # st.metric(
        #     label="Total Available Population",
        #     value=f"{total_population:,.0f}",
        #     help="""
        #     **How Calculated:** Sum of estimated populations across all facilities
            
        #     **Population Estimation Process:**
        #     1. Extract age/gender criteria from historical trials
        #     2. Query demographic APIs for regional population data
        #     3. Apply proportional matching for age group overlaps
        #     4. Aggregate across all demographic segments
            
        #     **Data Sources:**
        #     • US States: Census Bureau American Community Survey (ACS)
        #     • Countries: World Bank Population Statistics
            
        #     **Note:** Represents potential participant pool, not guaranteed enrollment
        #     """
        # )

def download_csv(results_df: pd.DataFrame, condition_name: str, target_country: str, target_state: str = None) -> None:
    """Create download button for CSV file"""
    
    if results_df.empty:
        return
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_condition = re.sub(r'[^\w\s-]', '', condition_name).strip().replace(' ', '_')
    safe_country = re.sub(r'[^\w\s-]', '', target_country).strip().replace(' ', '_')
    filename = f"clinical_trial_analysis_{safe_condition}_{safe_country}_{timestamp}.csv"
    
    # Convert dataframe to CSV
    csv_buffer = io.StringIO()
    results_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_data,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
        help="""
        **📁 CSV File Contents:**
        • All facility recommendations with complete data
        • Facility names, locations, and contact information
        • Suitability scores and ranking information
        • Population estimates and enrollment data
        • Demographic requirements (age/gender)
        • Study counts and historical data
        
        **📊 Data Sources Included:**
        • AACT Database: Facility and study information
        • Census/World Bank: Population demographics
        • Calculated Metrics: Suitability scores and estimates
        
        **💡 Use Cases:**
        • Further analysis in Excel/Python/R
        • Sharing with stakeholders
        • Site selection documentation
        • Recruitment planning
        """
    )

@st.fragment
def display_dataframe_with_pagination(df,key,items_per_page=1000):
    """Displays a Pandas DataFrame in Streamlit with pagination."""
    if len(df)>items_per_page:
        num_pages = (len(df) // items_per_page) + 1
        stc=st.columns([4,2], gap='large', vertical_alignment='bottom')
        with stc[0]:
            current_page = st.slider(f"Pagination: {items_per_page:,} records per page", min_value=1, max_value=num_pages, value=1,key=key)
        start_index = (current_page - 1) * items_per_page
        end_index = start_index + items_per_page
        st.dataframe(df[start_index:end_index])
    else:
         st.dataframe(df)


@st.fragment
def display_trial_data_analysis(trial_data: pd.DataFrame):
    with st.expander("Raw Clinical Trial Data"):
        display_dataframe_with_pagination(trial_data, "trial_data_pagination")
        
    trial_data_cp=trial_data.dropna(subset=['start_date','completion_date'])
    st.subheader("Existing Clinical Trial Metrics")
    first_trial = trial_data_cp.loc[trial_data_cp['start_date'].idxmin()]
    st.markdown(f"""First Trail Registration Date: **{pd.to_datetime(trial_data['start_date']).dropna().min().strftime('%Y-%m-%d')}** \t
    | Max Trail Completion Date: **{pd.to_datetime(trial_data['completion_date']).dropna().max().strftime('%Y-%m-%d')}**""")
    st.markdown(f"""First Trial Sponsor: **{first_trial['lead_sponsor']}** \t
    | First Trial Facility: **{first_trial['facility_name']} ({first_trial['facility_city']}, {first_trial['facility_country']})**""") 
    c= st.columns(5)
    c[0].metric("Total Unique Studies", trial_data['nct_id'].nunique())
    c[1].metric("Unique Sponsors", trial_data['lead_sponsor'].nunique())
    c[2].metric("Total Facilities", trial_data['facility_name'].nunique())
    c[3].metric("Total Intervention", trial_data['intervention_type'].nunique())
    c[4].metric("Unique Countries", trial_data['facility_country'].nunique())

    col1, col2, col3,col4 = st.columns(4)
    with col1:
        # Study status metrics
        st.markdown("**Study Status Distribution**")
        status_counts = trial_data.groupby('study_status')['nct_id'].nunique().reset_index()
        # status_counts
        st.bar_chart(status_counts,x='study_status',y='nct_id',horizontal=True)
    with col2:    
        # Sponsor type distribution
        st.markdown("**Sponsor Type Distribution**")
        sponsor_type_counts = trial_data.groupby('sponsor_type')['lead_sponsor'].nunique().reset_index()
        st.bar_chart(sponsor_type_counts,x='sponsor_type',y='lead_sponsor',horizontal=True)
    with col3:
        # Phase distribution
        st.markdown("**Trial Phase Distribution**")
        phase_counts = trial_data.groupby('phase')['nct_id'].nunique().reset_index()
        st.bar_chart(phase_counts,x='phase',y='nct_id',horizontal=True)
    with col4:
            # Intervention types
        st.markdown("**Intervention Type Distribution**")
        intervention_counts = trial_data['intervention_type'].value_counts()
        st.bar_chart(intervention_counts,horizontal=True)   
    

@st.cache_data(persist=True)
def get_condition_names():
    try:
        query = "SELECT DISTINCT downcase_name FROM conditions"
        conditions_df = st.session_state.db_manager.execute_query(query)
        return sorted(conditions_df['downcase_name'].unique())
    except Exception as e:
        st.error(f"Error loading conditions: {str(e)}")
        return []

@st.cache_data(persist=True)
def get_distinct_countries():
    try:
        query = "Select Distinct NAME from countries where name is not null"
        countries_df = st.session_state.db_manager.execute_query(query)
        return sorted(countries_df['name'].unique())
    except Exception as e:
        st.error(f"Error loading countries: {str(e)}")
        return []

@st.cache_data(persist=True)
def get_distinct_inv_types():
    try:
        query = "SElect distinct intervention_type from interventions"
        inv_types_df = st.session_state.db_manager.execute_query(query)
        return sorted(inv_types_df['intervention_type'].unique())
    except Exception as e:
        st.error(f"Error loading intervention types: {str(e)}")
        return []

def analysis_params():
    col1, col2, col3,col4 = st.columns([3,2,2,2],vertical_alignment='bottom')
    with col1:
        # Input fields
        # Lazy load condition names from AACT database
        # Load all conditions once using the cached function
        condition_name = st.selectbox(
                            "Medical Condition",
                            options=get_condition_names(),
                            index=None, # Prevents a default selection
                            placeholder="Search for a condition, e.g., Diabetes",
                            help="""
                            **Data Source:** AACT conditions table (from ClinicalTrials.gov)
                            
                            **Search Strategy:**
                            • Just start typing to filter the list below.
                            • Matches against historical completed studies only.
                            
                            **Examples:** Diabetes, Cancer, Hypertension, Alzheimer, COVID-19
                            """
        )
    with col2:
        target_country = st.selectbox(
            "Target Country",
            options=get_distinct_countries(),
            index=None, # Prevents a default selection
            help="""
            **What This Does:** Defines geographic region for population analysis
            
            **Data Sources:**
            • **US**: State-level data from Census Bureau API
            • **Other Countries**: National data from World Bank API
            
            **Population Data:**
            • Age/gender demographic breakdowns
            • Updated annually (World Bank) or every 5 years (Census)
            • Used to estimate available participant pools
            
            **Impact on Results:** Determines population estimates for suitability scoring
            """
        )
        target_state = None
    
    with col3:    
        intervention_type = st.selectbox(
            "Intervention Type",
            options=get_distinct_inv_types(),
            index=None, # Prevents a default selection
            help="""
                **What This Does:** Defines intervention type for further fileration of the analysis
                
                **Data Sources:**
                • **Intervention Type**: Intervention type from AACT database
                
                **Impact on Results:** Determines population estimates for suitability scoring and further fileration of the analysis
            """
        )
        
    
    # Advanced settings in expander
    with st.expander("⚙️ Advanced Settings"):
        adv_col1, adv_col2 = st.columns(2)
        
        with adv_col1:
            min_study_count = st.number_input(
                "Minimum Study Count",
                min_value=1,
                max_value=10,
                value=2,
                help="""
                **Purpose:** Ensures statistical reliability of enrollment estimates
                
                **How It Works:** Filters out facilities with too few historical studies
                
                **Data Source:** Count of completed studies per facility from AACT database
                
                **Recommendations:**
                • **2 (Default):** Good balance of coverage and reliability
                • **3-5:** Higher confidence, fewer results
                • **1:** Maximum coverage, less reliable averages
                
                **Impact:** Higher values = fewer but more reliable facility recommendations
                """
            )
        
        with adv_col2:
            score_threshold = st.number_input(
                "Minimum Suitability Score",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                help="""
                **Purpose:** Filter results to show only high-potential locations
                
                **Score Calculation:** Population ÷ Average Past Enrollment
                
                **Interpretation:**
                • **0 (Default):** Show all results
                • **10+:** Focus on moderate+ potential
                • **25+:** Focus on good+ potential  
                • **50+:** Focus on excellent potential only
                
                **Use Cases:**
                • Large studies: Set higher threshold (25-50)
                • Rare conditions: Keep lower threshold (0-10)
                • Competitive markets: Higher threshold for best sites
                """
            )
    
    # Analysis button
    with col4:
        run_analysis = st.button(
            "🚀 Run Analysis",
            type="primary",
            use_container_width=True
        )
    
    return condition_name, target_country, target_state,intervention_type, min_study_count, score_threshold, run_analysis

@st.fragment
def execute_analysis(condition_name, target_country, target_state, intervention_type, min_study_count, score_threshold):
    if not condition_name:
        st.error("❌ Please enter a medical condition.")
        return
    
    # Initialize optimizer with custom database settings
    optimizer = ClinicalTrialOptimizer()
    optimizer.db_connector = st.session_state.db_manager
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        progress_bar.progress(20)
        
        
        # Step 2: Fetch trial data
        status_text.text("📊 Fetching clinical trial data...")
        progress_bar.progress(40)
        
        trial_data = optimizer.get_trial_data(condition_name,target_country,intervention_type)
        
        if trial_data.empty:
            st.warning(f"⚠️ No completed trials found for condition: **{condition_name}**")
            st.info("Try using broader search terms (e.g., 'Diabetes' instead of 'Type 2 Diabetes')")
            return
        
        st.success(f"✅ Found {len(trial_data):,} trial records")
        display_trial_data_analysis(trial_data)
        
        # Step 3: Fetch population data and calculate scores
        status_text.text("🌍 Fetching population data and calculating suitability scores...")
        progress_bar.progress(70)
        

        if not condition_name or not target_country:
            st.warning("⚠️ Please select a target country for Feasibility Analysis.")
            return

        results = optimizer.calculate_suitability_scores(trial_data, target_country, target_state)
        
        if results.empty:
            st.warning("⚠️ No suitable locations found for the specified criteria.")
            return
        
        # Apply filters
        if score_threshold > 0:
            results = results[results['suitability_score'] >= score_threshold]
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis completed!")
        
        # Display results
        st.header("📈 Analysis Results")

        # Summary section
        st.subheader("📊 Summary")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.info(f"""
            **📊 Analysis Parameters:**
            - **Condition:** {condition_name}
            - **Target Region:** {target_country}{f', {target_state}' if target_state else ''}
            - **Study Records:** {len(trial_data):,} (AACT database)
            - **Locations Found:** {len(results)} (filtered by criteria)
            - **Min Study Count:** {min_study_count} studies per facility
            - **Score Threshold:** {score_threshold} minimum suitability
            
            **🔍 Data Sources:**
            - **Studies:** AACT database (ClinicalTrials.gov)
            - **Population:** {'US Census Bureau API' if target_state else 'World Bank API'}
            - **Processing:** Real-time demographic matching
            """)
        
        with col2:
            if len(results) > 0:
                top_location = results.iloc[0]
                st.success(f"""
                **🏆 Top Recommendation:**
                - **Facility:** {top_location['facility_name']}
                - **Location:** {top_location['facility_city']}, {top_location['facility_state']}
                - **Suitability Score:** {top_location['suitability_score']}
                - **Study Count:** {top_location['study_count']} historical studies
                - **Avg Enrollment:** {top_location['avg_past_enrollment']:.1f} participants
                - **Available Population:** {top_location['estimated_matching_population']:,}
                
                **💡 Score Meaning:** {top_location['estimated_matching_population']:,} ÷ {top_location['avg_past_enrollment']:.1f} = {top_location['suitability_score']:.1f}
                """)
        
        # Metrics
        display_summary_metrics(results)
        
        # Visualizations
        st.subheader("📊 Data Visualizations")
        create_visualizations(results)
        
        # Detailed results table
        st.subheader("📋 Detailed Results")
        
        # Column explanations
        with st.expander("ℹ️ Column Explanations & Data Sources", expanded=False):
            st.markdown("""
            ### 📊 **Column Definitions**
            
            | **Column** | **Description** | **Data Source** | **Calculation Method** |
            |------------|-----------------|-----------------|----------------------|
            | **Facility Name** | Research institution or hospital name | AACT Database - facilities table | Direct from study records |
            | **City/State/Country** | Geographic location of facility | AACT Database - facilities table | Direct from study records |
            | **Condition** | Medical condition from historical trials | AACT Database - conditions table | Matched with search criteria |
            | **Required Age Range** | Age eligibility from past studies | AACT Database - eligibilities table | Parsed from min/max age fields |
            | **Required Gender** | Gender criteria from past studies | AACT Database - eligibilities table | Direct from eligibility criteria |
            | **Avg Past Enrollment** | Average participants per study | AACT Database - studies table | Mean enrollment across completed studies |
            | **Study Count** | Number of historical studies | AACT Database - studies table | Count of completed studies per facility |
            | **Estimated Matching Population** | Available participants in region | US Census API / World Bank API | Age/gender demographic matching |
            | **Suitability Score** | Recruitment potential indicator | Calculated metric | Population ÷ Average Enrollment |
            
            ### 🔍 **Data Quality Notes**
            - **AACT Database**: Updated monthly from ClinicalTrials.gov
            - **Census Data**: American Community Survey 5-year estimates
            - **World Bank**: Annual population statistics by age/gender
            - **Calculations**: Real-time processing with proportional age matching
            """)
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender_filter = st.selectbox(
                "Filter by Gender",
                options=["All"] + list(results['required_gender'].unique()),
                help="Filter results by gender requirements from historical studies"
            )
        
        with col2:
            city_filter = st.selectbox(
                "Filter by City",
                options=["All"] + sorted(results['facility_city'].unique()),
                help="Filter results by facility city location"
            )
        
        with col3:
            top_n = st.number_input(
                "Show Top N Results",
                min_value=5,
                max_value=len(results),
                value=min(20, len(results)),
                help="Number of top-ranked facilities to display (ranked by suitability score)"
            )
        
        # Apply filters
        filtered_results = results.copy()
        if gender_filter != "All":
            filtered_results = filtered_results[filtered_results['required_gender'] == gender_filter]
        if city_filter != "All":
            filtered_results = filtered_results[filtered_results['facility_city'] == city_filter]
        
        filtered_results = filtered_results.head(top_n)
        
        # Add info about table formatting
        st.info("""
            **📊 Table Features:**
            - **Color Coding:** Darker green = higher suitability score
            - **Sorting:** Click column headers to sort data
            - **Ranking:** Results pre-sorted by suitability score (best first)
            - **Formatting:** Numbers formatted for readability (e.g., 1,234 for population)
        """)
        
        # Display table with formatting
        st.dataframe(
            filtered_results.style.format({
                'suitability_score': '{:.1f}',
                'avg_past_enrollment': '{:.1f}',
                'estimated_matching_population': '{:,.0f}'
            }).background_gradient(subset=['suitability_score'], cmap='viridis'),
            use_container_width=True,
            height=400
        )
        
        # Download section
        st.subheader("💾 Export Results")
        download_csv(results, condition_name, target_country, target_state)
        
        # Additional insights
        if len(results) > 0:
            st.subheader("💡 Key Insights")
            
            # Add explanation for insights
            st.info("""
            **📊 Insight Calculations:**
            • **Geographic Distribution:** Count of facilities per state/region from AACT data
            • **Demographic Patterns:** Historical study gender requirements from eligibility criteria
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📍 Geographic Distribution:**")
                st.caption("*Top 5 states/regions by facility count*")
                location_counts = results.groupby(['facility_state']).size().sort_values(ascending=False).head(5)
                for state, count in location_counts.items():
                    st.write(f"• {state}: {count} facilities")
                
                if len(location_counts) > 0:
                    st.markdown(f"**📊 Data Source:** AACT facilities table - {len(results)} total facilities analyzed")
            
            with col2:
                st.markdown("**👥 Demographic Patterns:**")
                st.caption("*Gender requirements from historical studies*")
                gender_dist = results['required_gender'].value_counts()
                for gender, count in gender_dist.items():
                    percentage = (count / len(results)) * 100
                    st.write(f"• {gender}: {count} studies ({percentage:.1f}%)")
                
                if len(gender_dist) > 0:
                    st.markdown(f"**📊 Data Source:** AACT eligibilities table - {len(results)} demographic groups analyzed")
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        logger.error(f"Analysis failed: {e}")
    
    finally:
        progress_bar.empty()
        status_text.empty()
        # optimizer.db_connector.disconnect()
            

def render_trail_feasibility_page ():
    """Main Streamlit application"""
    
    # Header
    st.header("🔬 Trial Feasibility and Site Selection",divider=True)
     # Check if database is connected
    if not check_database_connection():
        return
        
    st.markdown("""
    <div class="insight-box">
    <h5>🔍 Find optimal locations for clinical trials based on population demographics and historical data</h5>
    <p>Analyze historical clinical trial data from the AACT database and correlate it with 
    regional population demographics to identify the most suitable locations for future clinical trials.</p>
    </div>
    """, unsafe_allow_html=True)
 
    with st.container(border=True):
        st.subheader("📋 Analysis Parameters")
        # Create two columns
        condition_name, target_country, target_state,intervention_type, min_study_count, score_threshold, run_analysis=analysis_params()

     # Check if database is connected
    if not hasattr(st.session_state, 'database_loaded') or not st.session_state.database_loaded:
        st.warning("Please connect to a database from the Home page to view trail feasibility data.")
        return
    
    
    # Main content area
    if run_analysis:
        execute_analysis(condition_name, target_country, target_state, intervention_type, min_study_count, score_threshold)
    
    else:
        # Landing page content
        c=st.columns(2)
        with c[0]:
            st.markdown("""
            ## 🎯 How It Works
            
            1. **🔗 Database Setup**: Configure connection to AACT or custom database (optional)
            2. **📝 Enter Parameters**: Specify the medical condition and target region
            3. **🔍 Data Analysis**: The system queries historical clinical trial data
            4. **🌍 Population Matching**: Correlates trial demographics with regional population data
            5. **📊 Suitability Scoring**: Calculates location suitability based on available population vs. enrollment needs
            6. **📈 Results & Insights**: Provides ranked recommendations with interactive visualizations
             
            ## 💡 Example Use Cases
            
            - **Site Selection**: Identify optimal locations for Phase II/III trials
            - **Recruitment Planning**: Estimate patient availability in different regions
            - **Competitive Analysis**: Understand market saturation for specific conditions
            - **Strategic Planning**: Guide expansion into new therapeutic areas or regions
           
            """)
        with c[1]:
             st.markdown("""
             ## 📊 What You'll Get
            
            - **Ranked facility recommendations** with suitability scores
            - **Interactive visualizations** of the analysis results  
            - **Detailed demographic breakdowns** for each location
            - **Exportable CSV reports** for further analysis
            - **Key insights** about geographic and demographic patterns
            
            ## 🔗 Database Flexibility
            
            - **Default AACT**: Uses the public AACT database (recommended for most users)
            - **Custom PostgreSQL**: Connect to your organization's database instance
            - **Local Databases**: Use local copies for improved performance
            - **Test Connection**: Verify connectivity before running analysis
            - **Secure**: Password fields are masked for security
            """)
        st.markdown("""
        **💡 Pro Tip**: The default AACT settings work for most users. Only modify if you need to connect to a custom database instance.
        """)

        
        

render_trail_feasibility_page()