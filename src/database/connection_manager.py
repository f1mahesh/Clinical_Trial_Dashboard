"""
Database Connection Manager
==========================

Handles database connections for both sample and real AACT databases.
Provides a unified interface for database operations across the application.
"""

import streamlit as st
import pandas as pd
import sqlite3
import warnings
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import random

# Database connectivity
try:
    import psycopg2
    from sqlalchemy import create_engine, text
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False

warnings.filterwarnings('ignore')

class DatabaseConnectionManager:
    """Manages database connections and operations"""
    
    def __init__(self):
        """Initialize the database connection manager"""
        self.connection = None
        self.database_type = None
        self.engine = None
        
    def create_sample_database(self) -> bool:
        """
        Create and populate a sample SQLite database with enhanced synthetic clinical trial data.
        The sample data is more realistic, with richer relationships and more diverse values.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create SQLite database
            self.connection = sqlite3.connect(':memory:', check_same_thread=False)
            self.database_type = 'sample'
            cur = self.connection.cursor()

            # Create tables with more realistic schemas
            cur.execute("""
                CREATE TABLE studies (
                    nct_id TEXT PRIMARY KEY,
                    brief_title TEXT,
                    status TEXT,
                    start_date TEXT,
                    completion_date TEXT,
                    phase TEXT,
                    study_type TEXT,
                    enrollment INTEGER,
                    sponsor_id INTEGER,
                    is_fda_regulated_drug TEXT,
                    is_fda_regulated_device TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE sponsors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    agency_class TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE conditions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nct_id TEXT,
                    name TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE interventions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nct_id TEXT,
                    intervention_type TEXT,
                    name TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE facilities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nct_id TEXT,
                    name TEXT,
                    city TEXT,
                    state TEXT,
                    country TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nct_id TEXT,
                    outcome_type TEXT,
                    title TEXT,
                    time_frame TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE calculated_values (
                    nct_id TEXT PRIMARY KEY,
                    number_of_facilities INTEGER,
                    number_of_conditions INTEGER,
                    number_of_interventions INTEGER
                )
            """)
            cur.execute("""
                CREATE TABLE designs (
                    nct_id TEXT PRIMARY KEY,
                    allocation TEXT,
                    intervention_model TEXT,
                    masking TEXT,
                    primary_purpose TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE eligibilities (
                    nct_id TEXT PRIMARY KEY,
                    gender TEXT,
                    minimum_age TEXT,
                    maximum_age TEXT,
                    healthy_volunteers TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE investigators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nct_id TEXT,
                    name TEXT,
                    role TEXT,
                    affiliation TEXT
                )
            """)

            # Generate sample sponsors
            sponsor_names = [
                "PharmaGenix", "BioHealth Inc.", "MedInnovate", "Global Trials Org", "Academic Medical Center",
                "OncoResearch", "CardioLife", "NeuroSolutions", "ImmunoCore", "Pediatric Research Group"
            ]
            agency_classes = ["Industry", "NIH", "Other", "Academic"]
            for name in sponsor_names:
                cur.execute(
                    "INSERT INTO sponsors (name, agency_class) VALUES (?, ?)",
                    (name, random.choice(agency_classes))
                )
            self.connection.commit()

            # Fetch sponsor ids for assignment
            cur.execute("SELECT id, name FROM sponsors")
            sponsor_rows = cur.fetchall()
            sponsor_id_map = {row[1]: row[0] for row in sponsor_rows}

            # Generate sample studies
            statuses = ["Completed", "Recruiting", "Active, not recruiting", "Terminated", "Withdrawn"]
            phases = ["Phase 1", "Phase 2", "Phase 3", "Phase 4", "N/A"]
            study_types = ["Interventional", "Observational"]
            fda_drug = ["1", "0"]
            fda_device = ["1", "0"]

            n_studies = 100
            study_nct_ids = []
            for i in range(n_studies):
                nct_id = f"NCT{10000000 + i}"
                study_nct_ids.append(nct_id)
                sponsor_name = random.choice(sponsor_names)
                sponsor_id = sponsor_id_map[sponsor_name]
                status = random.choices(statuses, weights=[0.5, 0.2, 0.15, 0.1, 0.05])[0]
                phase = random.choice(phases)
                study_type = random.choice(study_types)
                enrollment = random.randint(30, 2000)
                start_date = (datetime(2015, 1, 1) + timedelta(days=random.randint(0, 3000))).strftime("%Y-%m-%d")
                completion_date = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=random.randint(180, 1200))).strftime("%Y-%m-%d")
                is_fda_drug = random.choices(fda_drug, weights=[0.3, 0.7])[0]
                is_fda_device = random.choices(fda_device, weights=[0.2, 0.8])[0]
                brief_title = f"{random.choice(['A Study of', 'Evaluation of', 'Trial of', 'Assessment of'])} {random.choice(['DrugX', 'TherapyY', 'DeviceZ', 'CompoundA', 'VaccineB'])} in {random.choice(['Cancer', 'Diabetes', 'Heart Disease', 'Asthma', 'Arthritis'])}"
                cur.execute(
                    "INSERT INTO studies (nct_id, brief_title, status, start_date, completion_date, phase, study_type, enrollment, sponsor_id, is_fda_regulated_drug, is_fda_regulated_device) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (nct_id, brief_title, status, start_date, completion_date, phase, study_type, enrollment, sponsor_id, is_fda_drug, is_fda_device)
                )
            self.connection.commit()

            # Generate sample conditions
            condition_names = [
                "Cancer", "Diabetes", "Heart Disease", "Asthma", "Arthritis", "Alzheimer's", "Obesity", "COVID-19", "Hypertension", "Depression"
            ]
            for nct_id in study_nct_ids:
                for _ in range(random.randint(1, 2)):
                    cond = random.choice(condition_names)
                    cur.execute("INSERT INTO conditions (nct_id, name) VALUES (?, ?)", (nct_id, cond))
            self.connection.commit()

            # Generate sample interventions
            intervention_types = ["Drug", "Device", "Procedure", "Behavioral", "Biological"]
            intervention_names = ["DrugX", "TherapyY", "DeviceZ", "CompoundA", "VaccineB", "ProcedureC", "BehaviorD"]
            for nct_id in study_nct_ids:
                for _ in range(random.randint(1, 2)):
                    itype = random.choice(intervention_types)
                    iname = random.choice(intervention_names)
                    cur.execute("INSERT INTO interventions (nct_id, intervention_type, name) VALUES (?, ?, ?)", (nct_id, itype, iname))
            self.connection.commit()

            # Generate sample facilities
            cities = ["New York", "Boston", "Chicago", "San Francisco", "Houston", "London", "Berlin", "Tokyo", "Sydney", "Toronto"]
            states = ["NY", "MA", "IL", "CA", "TX", "N/A", "N/A", "N/A", "N/A", "ON"]
            countries = ["USA", "USA", "USA", "USA", "USA", "UK", "Germany", "Japan", "Australia", "Canada"]
            for nct_id in study_nct_ids:
                for _ in range(random.randint(1, 3)):
                    idx = random.randint(0, len(cities)-1)
                    cur.execute(
                        "INSERT INTO facilities (nct_id, name, city, state, country) VALUES (?, ?, ?, ?, ?)",
                        (nct_id, f"{cities[idx]} Medical Center", cities[idx], states[idx], countries[idx])
                    )
            self.connection.commit()

            # Generate sample outcomes
            outcome_types = ["Primary", "Secondary"]
            outcome_titles = [
                "Overall Survival", "Progression-Free Survival", "HbA1c Reduction", "Blood Pressure Change", "Adverse Events", "Quality of Life"
            ]
            for nct_id in study_nct_ids:
                for _ in range(random.randint(1, 2)):
                    otype = random.choice(outcome_types)
                    otitle = random.choice(outcome_titles)
                    time_frame = f"{random.randint(6, 36)} months"
                    cur.execute(
                        "INSERT INTO outcomes (nct_id, outcome_type, title, time_frame) VALUES (?, ?, ?, ?)",
                        (nct_id, otype, otitle, time_frame)
                    )
            self.connection.commit()

            # Generate sample calculated values
            for nct_id in study_nct_ids:
                cur.execute("SELECT COUNT(*) FROM facilities WHERE nct_id=?", (nct_id,))
                n_fac = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM conditions WHERE nct_id=?", (nct_id,))
                n_cond = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM interventions WHERE nct_id=?", (nct_id,))
                n_int = cur.fetchone()[0]
                cur.execute(
                    "INSERT INTO calculated_values (nct_id, number_of_facilities, number_of_conditions, number_of_interventions) VALUES (?, ?, ?, ?)",
                    (nct_id, n_fac, n_cond, n_int)
                )
            self.connection.commit()

            # Generate sample designs
            allocations = ["Randomized", "Non-Randomized"]
            intervention_models = ["Parallel Assignment", "Crossover Assignment", "Single Group Assignment"]
            maskings = ["None", "Single", "Double", "Triple"]
            primary_purposes = ["Treatment", "Prevention", "Diagnostic", "Supportive Care"]
            for nct_id in study_nct_ids:
                allocation = random.choice(allocations)
                intervention_model = random.choice(intervention_models)
                masking = random.choice(maskings)
                primary_purpose = random.choice(primary_purposes)
                cur.execute(
                    "INSERT INTO designs (nct_id, allocation, intervention_model, masking, primary_purpose) VALUES (?, ?, ?, ?, ?)",
                    (nct_id, allocation, intervention_model, masking, primary_purpose)
                )
            self.connection.commit()

            # Generate sample eligibilities
            genders = ["All", "Male", "Female"]
            for nct_id in study_nct_ids:
                gender = random.choice(genders)
                min_age = f"{random.randint(0, 18)} Years"
                max_age = f"{random.randint(50, 90)} Years"
                healthy_volunteers = random.choice(["Accepts Healthy Volunteers", "Does Not Accept Healthy Volunteers"])
                cur.execute(
                    "INSERT INTO eligibilities (nct_id, gender, minimum_age, maximum_age, healthy_volunteers) VALUES (?, ?, ?, ?, ?)",
                    (nct_id, gender, min_age, max_age, healthy_volunteers)
                )
            self.connection.commit()

            # Generate sample investigators
            investigator_roles = ["Principal Investigator", "Sub-Investigator", "Study Director"]
            affiliations = [
                "Harvard Medical School", "Stanford University", "Mayo Clinic", "UCSF", "Cleveland Clinic",
                "Imperial College London", "Charité Berlin", "University of Tokyo", "University of Sydney", "UHN Toronto"
            ]
            for nct_id in study_nct_ids:
                for _ in range(random.randint(1, 2)):
                    name = f"Dr. {random.choice(['Smith', 'Johnson', 'Lee', 'Patel', 'Garcia', 'Kim', 'Chen', 'Singh', 'Brown', 'Martin'])}"
                    role = random.choice(investigator_roles)
                    affiliation = random.choice(affiliations)
                    cur.execute(
                        "INSERT INTO investigators (nct_id, name, role, affiliation) VALUES (?, ?, ?, ?)",
                        (nct_id, name, role, affiliation)
                    )
            self.connection.commit()

            # Update session state
            st.session_state.database_loaded = True
            st.session_state.database_type = 'sample'

            return True
            
        except Exception as e:
            st.error(f"Error creating sample database: {e}")
            return False
    
    def connect_to_local_database(self, host: str, port: int, database: str, 
                           username: str, password: str) -> bool:
        """
        Connect to real AACT PostgreSQL database
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            username: Database username
            password: Database password
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not POSTGRESQL_AVAILABLE:
            st.error("PostgreSQL support not available. Please install psycopg2 and sqlalchemy.")
            return False
        
        try:
            # Create connection string
           
            connection_string = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
           
            
            # print(connection_string)
            
            # Test connection
            engine = create_engine(connection_string)
            with engine.connect() as conn:
                test_query = "SELECT COUNT(*) as count FROM studies LIMIT 1"
                result = conn.execute(text(test_query))
                result.fetchone()
            
            # Store connection
            self.engine = engine
            self.database_type = 'local'
            
            # Update session state
            st.session_state.database_loaded = True
            st.session_state.database_type = 'local'
            st.session_state.connection_string = connection_string
            
            return True

        except Exception as e:
            st.error(f"Error connecting to Local database: {e}")
            return False

    def connect_to_real_aact(self, host: str, port: int, database: str, 
                           username: str, password: str) -> bool:
        """
        Connect to real AACT PostgreSQL database
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            username: Database username
            password: Database password
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not POSTGRESQL_AVAILABLE:
            st.error("PostgreSQL support not available. Please install psycopg2 and sqlalchemy.")
            return False
        
        try:
            # Create connection string
           
            connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
            
            # print(connection_string)
            
            # Test connection
            engine = create_engine(connection_string)
            with engine.connect() as conn:
                test_query = "SELECT COUNT(*) as count FROM studies LIMIT 1"
                result = conn.execute(text(test_query))
                result.fetchone()
            
            # Store connection
            self.engine = engine
            self.database_type = 'real'
            
            # Update session state
            st.session_state.database_loaded = True
            st.session_state.database_type = 'real'
            st.session_state.connection_string = connection_string
            return True
            
        except Exception as e:
            st.error(f"Error connecting to AACT database: {e}")
            return False
    
    def get_connection(self):
        """Get the current database connection"""
        if self.database_type == 'sample':
            return self.connection
        elif self.database_type == 'real':
            return self.engine
        elif self.database_type == 'local':
            return self.engine
        else:
            return None
    
    def execute_query(self, query: str, params: tuple = None) -> pd.DataFrame:
        """
        Execute a SQL query and return results as DataFrame
        
        Args:
            query: SQL query string
            params: Query parameters (optional)
            
        Returns:
            pd.DataFrame: Query results
        """
        try:
            if self.database_type == 'sample':
                if params:
                    return pd.read_sql(query, self.connection, params=params)
                else:
                    return pd.read_sql(query, self.connection)
            elif self.database_type == 'real':
                with self.engine.connect() as conn:
                    if params:
                        result = conn.execute(text(query), params)
                    else:
                        result = conn.execute(text(query))
                    return pd.DataFrame(result.fetchall(), columns=result.keys())
            elif self.database_type == 'local':
                with self.engine.connect() as conn:
                    if params:
                        result = conn.execute(text(query), params)
                    else:
                        result = conn.execute(text(query))
                    return pd.DataFrame(result.fetchall(), columns=result.keys())
            else:
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Error executing query: {e}")
            return pd.DataFrame()
    
    def _create_sample_studies(self):
        """Create sample studies table"""
        studies_data = []
        for i in range(1000):
            studies_data.append({
                'nct_id': f'NCT{str(i+1).zfill(8)}',
                'brief_title': f'Sample Study {i+1}',
                'official_title': f'Official Title for Sample Study {i+1}',
                'phase': random.choice(['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Not Applicable']),
                'overall_status': random.choice(['Completed', 'Recruiting', 'Active, not recruiting', 'Terminated', 'Suspended']),
                'study_type': random.choice(['Interventional', 'Observational']),
                'enrollment': random.randint(10, 10000),
                'source_class': random.choice(['Industry', 'NIH', 'Other', 'Other U.S. Federal']),
                'start_date': datetime.now() - timedelta(days=random.randint(0, 1000)),
                'completion_date': datetime.now() + timedelta(days=random.randint(0, 1000))
            })
        
        studies_df = pd.DataFrame(studies_data)
        studies_df.to_sql('studies', self.connection, if_exists='replace', index=False)
    
    def _create_sample_sponsors(self):
        """Create sample sponsors table"""
        sponsors_data = []
        studies_df = pd.read_sql('SELECT nct_id FROM studies', self.connection)
        
        for _, row in studies_df.iterrows():
            sponsors_data.append({
                'nct_id': row['nct_id'],
                'agency_class': random.choice(['Industry', 'NIH', 'Other', 'Other U.S. Federal']),
                'lead_or_collaborator': random.choice(['lead', 'collaborator'])
            })
        
        sponsors_df = pd.DataFrame(sponsors_data)
        sponsors_df.to_sql('sponsors', self.connection, if_exists='replace', index=False)
    
    def _create_sample_conditions(self):
        """Create sample conditions table"""
        conditions_data = []
        studies_df = pd.read_sql('SELECT nct_id FROM studies', self.connection)
        condition_names = ['Diabetes', 'Cancer', 'Heart Disease', 'Alzheimer\'s', 'COVID-19', 
                          'Hypertension', 'Asthma', 'Depression', 'Obesity', 'Arthritis']
        
        for _, row in studies_df.iterrows():
            conditions_data.append({
                'nct_id': row['nct_id'],
                'name': random.choice(condition_names)
            })
        
        conditions_df = pd.DataFrame(conditions_data)
        conditions_df.to_sql('conditions', self.connection, if_exists='replace', index=False)
    
    def _create_sample_interventions(self):
        """Create sample interventions table"""
        interventions_data = []
        studies_df = pd.read_sql('SELECT nct_id FROM studies LIMIT 500', self.connection)  # Sample subset
        
        intervention_types = ['Drug', 'Device', 'Biological', 'Behavioral', 'Procedure', 'Other']
        intervention_names = ['Test Drug A', 'Test Device B', 'Biological Agent C', 'Behavioral Therapy D']
        
        for _, row in studies_df.iterrows():
            interventions_data.append({
                'nct_id': row['nct_id'],
                'intervention_type': random.choice(intervention_types),
                'intervention_name': random.choice(intervention_names)
            })
        
        interventions_df = pd.DataFrame(interventions_data)
        interventions_df.to_sql('interventions', self.connection, if_exists='replace', index=False)
    
    def _create_sample_facilities(self):
        """Create sample facilities table"""
        facilities_data = []
        studies_df = pd.read_sql('SELECT nct_id FROM studies LIMIT 600', self.connection)
        
        countries = ['United States', 'Canada', 'United Kingdom', 'Germany', 'France', 'Japan', 'Australia']
        cities = ['New York', 'London', 'Toronto', 'Berlin', 'Paris', 'Tokyo', 'Sydney']
        
        for _, row in studies_df.iterrows():
            facilities_data.append({
                'nct_id': row['nct_id'],
                'name': f'Research Center {random.randint(1, 100)}',
                'country': random.choice(countries),
                'city': random.choice(cities)
            })
        
        facilities_df = pd.DataFrame(facilities_data)
        facilities_df.to_sql('facilities', self.connection, if_exists='replace', index=False)
    
    def _create_sample_outcomes(self):
        """Create sample outcomes table"""
        outcomes_data = []
        studies_df = pd.read_sql('SELECT nct_id FROM studies', self.connection)
        
        for _, row in studies_df.iterrows():
            outcomes_data.append({
                'nct_id': row['nct_id'],
                'outcome_type': random.choice(['Primary', 'Secondary']),
                'measure': f'Outcome Measure {random.randint(1, 5)}',
                'description': f'Description for outcome measure {random.randint(1, 5)}'
            })
        
        outcomes_df = pd.DataFrame(outcomes_data)
        outcomes_df.to_sql('outcomes', self.connection, if_exists='replace', index=False)
    
    def _create_sample_calculated_values(self):
        """Create sample calculated values table"""
        calculated_data = []
        studies_df = pd.read_sql('SELECT nct_id FROM studies', self.connection)
        
        for _, row in studies_df.iterrows():
            calculated_data.append({
                'nct_id': row['nct_id'],
                'actual_duration': random.randint(6, 60),
                'were_results_reported': random.choice(['0', '1']),
                'months_to_report_results': random.randint(1, 24)
            })
        
        calculated_df = pd.DataFrame(calculated_data)
        calculated_df.to_sql('calculated_values', self.connection, if_exists='replace', index=False)
    
    def _create_sample_designs(self):
        """Create sample designs table"""
        designs_data = []
        studies_df = pd.read_sql('SELECT nct_id FROM studies WHERE study_type = "Interventional" LIMIT 700', self.connection)
        
        for _, row in studies_df.iterrows():
            designs_data.append({
                'nct_id': row['nct_id'],
                'allocation': random.choice(['Randomized', 'Non-Randomized']),
                'intervention_model': random.choice(['Parallel Assignment', 'Crossover Assignment', 'Factorial Assignment']),
                'primary_purpose': random.choice(['Treatment', 'Prevention', 'Diagnostic', 'Supportive Care'])
            })
        
        designs_df = pd.DataFrame(designs_data)
        designs_df.to_sql('designs', self.connection, if_exists='replace', index=False)
    
    def _create_sample_eligibilities(self):
        """Create sample eligibilities table"""
        eligibilities_data = []
        studies_df = pd.read_sql('SELECT nct_id FROM studies LIMIT 800', self.connection)
        
        for _, row in studies_df.iterrows():
            eligibilities_data.append({
                'nct_id': row['nct_id'],
                'gender': random.choice(['All', 'Male', 'Female']),
                'age_group': random.choice(['All Ages', 'Child (birth-17)', 'Adult (18-64)', 'Older Adult (65+)']),
                'minimum_age': random.randint(0, 80),
                'maximum_age': random.randint(18, 100)
            })
        
        eligibilities_df = pd.DataFrame(eligibilities_data)
        eligibilities_df.to_sql('eligibilities', self.connection, if_exists='replace', index=False)
    
    def _create_sample_investigators(self):
        """Create sample investigators table"""
        investigators_data = []
        studies_df = pd.read_sql('SELECT nct_id FROM studies', self.connection)
        
        investigator_names = ['Dr. Smith', 'Dr. Johnson', 'Dr. Williams', 'Dr. Brown', 'Dr. Jones', 
                            'Dr. Garcia', 'Dr. Miller', 'Dr. Davis', 'Dr. Rodriguez', 'Dr. Martinez']
        
        for _, row in studies_df.iterrows():
            investigators_data.append({
                'nct_id': row['nct_id'],
                'name': random.choice(investigator_names),
                'role': random.choice(['Principal Investigator', 'Sub-Investigator', 'Study Director'])
            })
        
        investigators_df = pd.DataFrame(investigators_data)
        investigators_df.to_sql('investigators', self.connection, if_exists='replace', index=False) 