import streamlit as st
import pandas as pd
import psycopg2
import os
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import traceback
import re

# Streamlit Float for floating elements
from streamlit_float import *

# LangChain imports
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.schema import SystemMessage, HumanMessage
from langchain.tools import Tool
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferWindowMemory

# Google Gemini imports
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Set page configuration
# st.set_page_config(
#     page_title="Clinical Trials Query Agent",
#     page_icon="🔬",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# Custom CSS for better UI including chat widget
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        background-color: #f8f9fa;
    }
    .sql-code {
        background-color: #f1f3f4;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
    }
    .error-box {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #f44336;
    }
    .success-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #4caf50;
    }

</style>


""", unsafe_allow_html=True)

class AACTSchemaProcessor:
    """Process AACT schema information from CSV file"""
    
    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path
        self.schema_info = {}
        self.load_schema()
    
    def load_schema(self):
        """Load and process schema information from CSV"""
        try:
            # Try to read CSV with different approaches
            df = None
            
            # First, try with pipe delimiter
            try:
                df = pd.read_csv(self.csv_file_path, delimiter='|')
                st.info("📁 Schema loaded with pipe delimiter")
            except Exception as e:
                st.warning(f"Pipe delimiter failed: {str(e)}")
                
                # Fallback: try automatic delimiter detection
                try:
                    df = pd.read_csv(self.csv_file_path, sep=None, engine='python')
                    st.info("📁 Schema loaded with auto-detected delimiter")
                except Exception as e2:
                    st.error(f"Auto-detection failed: {str(e2)}")
                    
                    # Last resort: try comma delimiter
                    df = pd.read_csv(self.csv_file_path, delimiter=',')
                    st.info("📁 Schema loaded with comma delimiter")
            
            if df is None:
                raise ValueError("Could not read CSV file with any delimiter")
            
            # Clean the data - remove any rows where Table is NaN or empty
            df = df.dropna(subset=['Table'])
            df = df[df['Table'].str.strip() != '']
            
            # Group by table name
            for _, row in df.iterrows():
                table_name = str(row['Table']).strip()
                
                # Skip if table name is empty after stripping
                if not table_name or table_name.lower() == 'nan':
                    continue
                    
                if table_name not in self.schema_info:
                    self.schema_info[table_name] = {
                        'columns': [],
                        'description': f"Table containing {table_name} information"
                    }
                
                # Clean and extract column information
                column_info = {
                    'name': str(row.get('Field', '')).strip(),
                    'type': str(row.get('Type', '')).strip(),
                    'nullable': str(row.get('Nullable', 'Yes')).strip(),
                    'description': str(row.get('Description', '')).strip(),
                    'ctgov_data_point': str(row.get('CTGov Data Point', '')).strip(),
                    'api_field_path': str(row.get('CTGov API Field Path', '')).strip()
                }
                
                # Only add if we have a valid field name
                if column_info['name'] and column_info['name'].lower() != 'nan':
                    self.schema_info[table_name]['columns'].append(column_info)
            
            # Log successful loading
            total_tables = len(self.schema_info)
            total_columns = sum(len(table['columns']) for table in self.schema_info.values())
            st.success(f"✅ Schema loaded: {total_tables} tables, {total_columns} columns")
                
        except Exception as e:
            st.error(f"Error loading schema: {str(e)}")
            st.error("Please check that the documentation_20250730.csv file is in the correct format.")
            self.schema_info = {}
    
    def get_schema_summary(self) -> str:
        """Generate a comprehensive schema summary for the LLM"""
        summary = """
# AACT (Aggregate Analysis of ClinicalTrials.gov) Database Schema

## Overview
The AACT database contains comprehensive information about clinical trials from ClinicalTrials.gov.
The main identifier across tables is `nct_id` (National Clinical Trial identifier).

## Key Tables and Relationships:

### Core Tables:
"""
        
        # Identify core tables
        core_tables = ['studies', 'conditions', 'interventions', 'outcomes', 'facilities', 'sponsors']
        
        for table in core_tables:
            if table in self.schema_info:
                summary += f"\n**{table}:**\n"
                summary += f"- {self.schema_info[table]['description']}\n"
                
                # Add key columns
                key_columns = []
                for col in self.schema_info[table]['columns'][:10]:  # First 10 columns
                    if col['name'] in ['nct_id', 'id', 'name', 'title', 'type', 'status', 'phase', 'start_date', 'completion_date']:
                        key_columns.append(f"  - `{col['name']}` ({col['type']}): {col['description']}")
                
                if key_columns:
                    summary += "Key columns:\n" + "\n".join(key_columns) + "\n"
        
        summary += """
### Important Join Patterns:
- Most tables join to `studies` via `nct_id`
- Use `studies.nct_id` as the primary key for joins
- Common filters: phase, status, start_date, completion_date, country

### Query Best Practices:
1. Always use `nct_id` for joining tables
2. Use ILIKE for case-insensitive text matching
3. Date comparisons should use proper date format (YYYY-MM-DD)
4. Common aggregations: COUNT(), AVG(), SUM() for statistical analysis
5. Use appropriate WHERE clauses to filter by study status, phase, dates

### Sample Query Patterns:
```sql
-- Basic study count by phase
SELECT phase, COUNT(*) as study_count 
FROM studies 
WHERE phase IS NOT NULL 
GROUP BY phase;

-- Studies with conditions (join pattern)
SELECT s.nct_id, s.brief_title, c.name as condition
FROM studies s
JOIN conditions c ON s.nct_id = c.nct_id
WHERE c.name ILIKE '%cancer%';
```
"""
        
        return summary
    
    def get_table_info(self, table_name: str) -> Dict:
        """Get detailed information about a specific table"""
        return self.schema_info.get(table_name, {})

class ClinicalTrialsAgent:
    """RAG-powered agent for querying clinical trials database"""
    
    def __init__(self, db_connection_string: str, schema_processor: AACTSchemaProcessor, 
                 llm_provider: str = "openai", model_name: str = "gpt-4"):
        self.db_connection_string = db_connection_string
        self.schema_processor = schema_processor
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.db = None
        self.agent = None
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=5,
            return_messages=True
        )
        self.initialize_agent()
    
    def _create_llm(self):
        """Create the appropriate LLM based on provider selection"""
        if self.llm_provider == "openai":
            return ChatOpenAI(
                model_name=self.model_name,
                temperature=0,
                openai_api_key=st.session_state.get('openai_api_key', os.getenv('OPENAI_API_KEY'))
            )
        elif self.llm_provider == "gemini" and GEMINI_AVAILABLE:
            gemini_api_key = st.session_state.get('gemini_api_key', os.getenv('GEMINI_API_KEY'))
            if not gemini_api_key:
                raise ValueError("Gemini API key is required but not provided")
            
            # Configure Gemini
            genai.configure(api_key=gemini_api_key)
            
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=0,
                google_api_key=gemini_api_key
            )
        elif self.llm_provider == "custom":
            custom_api_key = st.session_state.get('custom_api_key')
            custom_base_url = st.session_state.get('custom_base_url')
            
            if not custom_api_key or not custom_base_url:
                raise ValueError("Custom LLM requires both API key and base URL")
            
            return ChatOpenAI(
                model_name=self.model_name,
                temperature=0,
                openai_api_key=custom_api_key,
                base_url=custom_base_url
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def initialize_agent(self):
        """Initialize the SQL agent with schema knowledge"""
        try:
            # Initialize database connection
            self.db = SQLDatabase.from_uri(self.db_connection_string)
            
            # Create LLM
            llm = self._create_llm()
            
            # Create toolkit
            toolkit = SQLDatabaseToolkit(db=self.db, llm=llm)
            
            # Enhanced system message with schema information
            system_message = f"""You are a helpful AI assistant specialized in querying the AACT clinical trials database.

{self.schema_processor.get_schema_summary()}

## Your Role:
1. Understand user questions about clinical trials
2. Generate accurate SQL queries using the AACT schema
3. Execute queries and interpret results
4. Provide clear, comprehensive answers

## Important Guidelines:
- Always use `nct_id` for joins between tables
- Use ILIKE for case-insensitive text searches
- Be precise with date formats and comparisons
- Provide context and explanation with your answers
- If a query fails, suggest alternative approaches

## SQL Query Guidelines:
- Generate ONE complete, well-formed SQL query per request
- Always format SQL queries within proper code blocks: ```sql ... ```
- Ensure the SQL query is complete and executable
- Do not repeat or fragment SQL statements
- Include proper semicolons at the end of statements

## Response Format:
When answering, provide:
1. A natural language summary of the findings
2. The complete SQL query used (formatted in a single code block)
3. Key insights from the data presented in a clear tabular format when applicable
4. Visual representations (charts, graphs) of the data when meaningful, using:
   - Bar charts for comparing categories
   - Line graphs for temporal trends
   - Pie charts for proportional data
   - Scatter plots for correlations
5. Statistical highlights and trends identified in the data

Remember: You're helping researchers and clinicians understand clinical trial data, so accuracy and clarity are paramount.
"""
            
            # Create agent
            self.agent = create_sql_agent(
                llm=llm,
                toolkit=toolkit,
                verbose=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                memory=self.memory,
                agent_executor_kwargs={
                    "memory": self.memory,
                    "handle_parsing_errors": True
                }
            )
            
            # Update system message
            if hasattr(self.agent, 'agent'):
                if hasattr(self.agent.agent, 'llm_chain'):
                    if hasattr(self.agent.agent.llm_chain, 'prompt'):
                        # Add our schema info to the existing prompt
                        original_template = self.agent.agent.llm_chain.prompt.template
                        enhanced_template = system_message + "\n\n" + original_template
                        self.agent.agent.llm_chain.prompt.template = enhanced_template
                        
        except Exception as e:
            st.error(f"Error initializing agent: {str(e)}")
            self.agent = None
    
    def query(self, question: str, callback_handler=None) -> Dict[str, Any]:
        """Execute a query using the agent"""
        if not self.agent:
            return {"error": "Agent not initialized"}
        
        try:
            if callback_handler:
                response = self.agent.run(question, callbacks=[callback_handler])
            else:
                response = self.agent.run(question)
            
            # Try to extract SQL queries and results from the response
            sql_queries = self._extract_sql_from_response(response)
            
            return {
                "success": True,
                "response": response,
                "sql_queries": sql_queries,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }
    
    def _extract_sql_from_response(self, response: str) -> List[str]:
        """Extract SQL queries from agent response"""
        
        # Priority patterns - more specific patterns first
        sql_patterns = [
            # Complete SQL code blocks (highest priority)
            r'```sql\s*\n(.*?)\n\s*```',
            # Generic code blocks containing SELECT
            r'```\s*\n(SELECT.*?)\n\s*```',
            # Inline SQL patterns
            r'Query:\s*(SELECT.*?)(?:\n\n|\n[A-Z]|\n$|$)',
            r'SQL:\s*(SELECT.*?)(?:\n\n|\n[A-Z]|\n$|$)',
        ]
        
        extracted_queries = []
        
        # Try each pattern in order of priority
        for pattern in sql_patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            for match in matches:
                cleaned = self._clean_sql_query(match)
                if cleaned and self._is_valid_sql(cleaned):
                    extracted_queries.append(cleaned)
            
            # If we found good matches with high priority patterns, don't try lower priority ones
            if extracted_queries:
                break
        
        # Remove duplicates while preserving order
        unique_queries = []
        seen = set()
        for query in extracted_queries:
            # Normalize query for duplicate detection
            normalized = re.sub(r'\s+', ' ', query.strip().upper())
            if normalized not in seen:
                seen.add(normalized)
                unique_queries.append(query)

        return unique_queries
    
    def _clean_sql_query(self, query: str) -> str:
        """Clean and normalize a SQL query"""
        if not query:
            return ""
        
        # Remove extra whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', query.strip())
        
        # Remove trailing semicolons if multiple exist
        cleaned = re.sub(r';+$', ';', cleaned)
        
        # Add semicolon if missing
        if not cleaned.endswith(';'):
            cleaned += ';'
        
        return cleaned
    
    def _is_valid_sql(self, query: str) -> bool:
        """Basic validation to check if the query looks like valid SQL"""
        if not query or len(query.strip()) < 10:
            return False
        
        query_upper = query.upper().strip()
        
        # Must start with SELECT, INSERT, UPDATE, DELETE, WITH, etc.
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH', 'CREATE', 'DROP', 'ALTER']
        starts_with_sql = any(query_upper.startswith(keyword) for keyword in sql_keywords)
        
        if not starts_with_sql:
            return False
        
        # Basic structure checks for SELECT queries
        if query_upper.startswith('SELECT'):
            # Must have FROM clause for most practical queries
            if 'FROM' not in query_upper:
                return False
            
            # Check for basic SQL structure (avoid fragments)
            essential_parts = query_upper.count('SELECT') + query_upper.count('FROM')
            if essential_parts < 2:  # At least one SELECT and one FROM
                return False
        
        # Avoid queries that are clearly fragments or duplicated content
        lines = query.strip().split('\n')
        if len(lines) > 1:
            # Check for repeated lines (sign of duplication)
            unique_lines = set(line.strip().upper() for line in lines if line.strip())
            if len(unique_lines) < len([line for line in lines if line.strip()]) * 0.7:
                return False  # Too many duplicate lines
        
        return True

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    
    if 'schema_processor' not in st.session_state:
        st.session_state.schema_processor = None
    
    if 'db_connected' not in st.session_state:
        st.session_state.db_connected = False
    
    if 'database_loaded' not in st.session_state:
        st.session_state.database_loaded = False
    
    if 'llm_provider' not in st.session_state:
        st.session_state.llm_provider = 'openai'
    
    if 'model_name' not in st.session_state:
        st.session_state.model_name = 'gpt-4'
    
    if 'chat_widget_open' not in st.session_state:
        st.session_state.chat_widget_open = False
    
    if 'show_config' not in st.session_state:
        st.session_state.show_config = False
    
    if 'custom_api_key' not in st.session_state:
        st.session_state.custom_api_key = None
    
    if 'custom_base_url' not in st.session_state:
        st.session_state.custom_base_url = None
    
    if 'custom_model_name' not in st.session_state:
        st.session_state.custom_model_name = 'cts-vibecode-gpt-4.1'

def setup_sidebar():
    """Setup the configuration sidebar"""
    st.sidebar.title("🔧 Configuration")
    
    # Database connection controls
    st.sidebar.subheader("📊 Database Connection")
    
    # Connection test button
    if st.sidebar.button("Test Database Connection", key="sidebar_test_connection_btn", type="primary"):
        test_connection()
    
    # Schema diagnostics
    st.sidebar.subheader("🔍 Schema Diagnostics")
    if st.sidebar.button("Show Schema Info", key="sidebar_schema_info_btn"):
        if st.session_state.schema_processor and st.session_state.schema_processor.schema_info:
            schema_info = st.session_state.schema_processor.schema_info
            st.sidebar.write(f"**Tables loaded:** {len(schema_info)}")
            
            # Show table names in expander
            with st.sidebar.expander("Table List"):
                for table_name in sorted(schema_info.keys()):
                    col_count = len(schema_info[table_name]['columns'])
                    st.write(f"• **{table_name}** ({col_count} columns)")
        else:
            st.sidebar.error("No schema information loaded")
    
    # Clear conversation button
    if st.sidebar.button("Clear Conversation", key="sidebar_clear_conversation_btn"):
        st.session_state.messages = []
        if st.session_state.agent and hasattr(st.session_state.agent, 'memory'):
            st.session_state.agent.memory.clear()
        st.rerun()

def test_connection():
    """Test database connection"""
    try:
        connection_string = st.session_state.connection_string
        
        # Test connection
        db = SQLDatabase.from_uri(connection_string)
        tables = db.get_usable_table_names()
        
        st.success(f"✅ Connected! Found {len(tables)} tables.")
        st.session_state.db_connected = True
        st.session_state.database_loaded = True
        
        # Initialize schema processor and agent
        if not st.session_state.schema_processor:
            st.session_state.schema_processor = AACTSchemaProcessor("config/AACT_Schema.csv")
        
        # Initialize agent only if we have API key
        llm_provider = st.session_state.get('llm_provider', 'openai')
        api_key_available = False
        
        if llm_provider == 'openai' and st.session_state.get('openai_api_key'):
            api_key_available = True
        elif llm_provider == 'gemini' and st.session_state.get('gemini_api_key'):
            api_key_available = True
        elif llm_provider == 'custom' and st.session_state.get('custom_api_key') and st.session_state.get('custom_base_url'):
            api_key_available = True
            
        if api_key_available and not st.session_state.agent:
            st.session_state.agent = ClinicalTrialsAgent(
                connection_string, 
                st.session_state.schema_processor,
                llm_provider=st.session_state.llm_provider,
                model_name=st.session_state.model_name
            )
    except Exception as e:
        st.error(f"❌ Connection failed: {str(e)}")
        st.session_state.db_connected = False

@st.dialog("LLM Configuration")
def render_llm_configuration():
    """Render LLM configuration in chat popup"""
    st.markdown("#### 🤖 AI Configuration")
    
    # LLM Provider Selection
    provider_options = ["OpenAI", "Custom LLM"]
    if GEMINI_AVAILABLE:
        provider_options.insert(1, "Google Gemini")
    
    current_provider_index = 0
    if st.session_state.get('llm_provider') == 'gemini':
        current_provider_index = 1 if GEMINI_AVAILABLE else 0
    elif st.session_state.get('llm_provider') == 'custom':
        current_provider_index = len(provider_options) - 1
    
    llm_provider = st.selectbox(
        "Choose LLM Provider",
        provider_options,
        index=current_provider_index,
        key="popup_llm_provider_selectbox",
        help="Select your preferred language model provider"
    )
    
    # Store provider selection
    if llm_provider == "OpenAI":
        provider_key = "openai"
    elif llm_provider == "Google Gemini":
        provider_key = "gemini"
    else:  # Custom LLM
        provider_key = "custom"
    
    st.session_state.llm_provider = provider_key
    
    # API Key configuration based on provider
    if llm_provider == "OpenAI":
        # Check if API key exists in session
        has_openai_key = bool(st.session_state.get('openai_api_key'))
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if has_openai_key:
                st.success("✅ OpenAI API Key configured")
            else:
                col1, col2 = st.columns([2, 1],vertical_alignment='bottom')
                with col1:
                    api_key = st.text_input(
                        "OpenAI API Key",
                        type="password",
                        key="popup_openai_api_key_input",
                        help="Enter your OpenAI API key for GPT-4 access",
                        placeholder="sk-..."
                    )
                with col2:
                    openai_confirm=st.button('Confirm',key='popup_openai_api_key_confirm_btn')
                
                if api_key and openai_confirm:
                    st.session_state.openai_api_key = api_key
                    os.environ['OPENAI_API_KEY'] = api_key
                    st.success("✅ API Key saved!")
                    test_connection()
                    # st.rerun()
        
        with col2:
            if has_openai_key:
                if st.button("🗑️ Clear", key="clear_openai_key_btn", help="Clear API Key"):
                    if 'openai_api_key' in st.session_state:
                        del st.session_state.openai_api_key
                    if 'OPENAI_API_KEY' in os.environ:
                        del os.environ['OPENAI_API_KEY']
                    st.session_state.agent = None  # Reset agent
                    st.success("API Key cleared!")
                    st.rerun()
        
        # Model selection for OpenAI
        if has_openai_key or not has_openai_key:  # Always show model selection for OpenAI
            openai_models = {
                "GPT-4 Models": [
                    "gpt-4",
                    "gpt-4-turbo",
                    "gpt-4-turbo-preview", 
                    "gpt-4-0125-preview",
                    "gpt-4-1106-preview"
                ],
                "GPT-3.5 Models": [
                    "gpt-3.5-turbo",
                    "gpt-3.5-turbo-0125",
                    "gpt-3.5-turbo-1106"
                ],
                "Custom/Azure Models": [
                    "cts-vibecode-gpt-4.1",
                    "custom-model"
                ]
            }
            
            # Flatten the model options for selectbox
            all_openai_models = []
            model_categories = []
            for category, models in openai_models.items():
                model_categories.extend([f"--- {category} ---"] + models)
                all_openai_models.extend(models)
            
            # Find current model index
            current_model = st.session_state.get('model_name', 'gpt-4')
            current_index = 0
            
            # Create display options (with category headers)
            display_options = []
            selectable_indices = []
            
            for i, item in enumerate(model_categories):
                if item.startswith("---"):
                    display_options.append(item)
                else:
                    display_options.append(item)
                    selectable_indices.append(len(display_options) - 1)
                    if item == current_model:
                        current_index = len(selectable_indices) - 1
            
            # Use a simple selectbox with just the models
            selected_model = st.selectbox(
                "OpenAI Model",
                all_openai_models,
                index=all_openai_models.index(current_model) if current_model in all_openai_models else 0,
                key="popup_openai_model_selectbox",
                help="Choose the OpenAI model to use"
            )
            st.session_state.model_name = selected_model
        
    elif llm_provider == "Google Gemini" and GEMINI_AVAILABLE:
        # Check if API key exists in session
        has_gemini_key = bool(st.session_state.get('gemini_api_key'))
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if has_gemini_key:
                st.success("✅ Gemini API Key configured")
            else:
                col1, col2 = st.columns([2, 1],vertical_alignment='bottom')
                with col1:
                    gemini_key = st.text_input(
                        "Gemini API Key",
                        type="password",
                        key="popup_gemini_api_key_input",
                        help="Enter your Google Gemini API key",
                        placeholder="AI..."
                    )
                with col2:
                    gemini_confirm=st.button('Confirm',key='popup_gemini_api_key_confirm_btn')
                
                if gemini_key and gemini_confirm:
                    st.session_state.gemini_api_key = gemini_key
                    os.environ['GEMINI_API_KEY'] = gemini_key
                    st.success("✅ API Key saved!")
                    test_connection()
                    # st.rerun()
        
        with col2:
            if has_gemini_key:
                if st.button("🗑️ Clear", key="clear_gemini_key_btn", help="Clear API Key"):
                    if 'gemini_api_key' in st.session_state:
                        del st.session_state.gemini_api_key
                    if 'GEMINI_API_KEY' in os.environ:
                        del os.environ['GEMINI_API_KEY']
                    st.session_state.agent = None  # Reset agent
                    st.success("API Key cleared!")
                    st.rerun()
        
        # Model selection for Gemini
        if has_gemini_key or not has_gemini_key:  # Always show model selection for Gemini
            gemini_models = {
                "Gemini 1.5 Models": [
                    "gemini-1.5-flash",
                    "gemini-1.5-flash-001",
                    "gemini-1.5-flash-002",
                    "gemini-1.5-pro",
                    "gemini-1.5-pro-001",
                    "gemini-1.5-pro-002"
                ],
                "Gemini 1.0 Models": [
                    "gemini-1.0-pro",
                    "gemini-1.0-pro-001",
                    "gemini-1.0-pro-vision-latest"
                ],
                "Experimental Models": [
                    "gemini-1.5-flash-8b",
                    "gemini-1.5-flash-8b-001"
                ]
            }
            
            # Flatten the model options
            all_gemini_models = []
            for category, models in gemini_models.items():
                all_gemini_models.extend(models)
            
            current_model = st.session_state.get('model_name', 'gemini-1.5-flash')
            
            selected_model = st.selectbox(
                "Gemini Model",
                all_gemini_models,
                index=all_gemini_models.index(current_model) if current_model in all_gemini_models else 0,
                key="popup_gemini_model_selectbox",
                help="Choose the Gemini model to use"
            )
            st.session_state.model_name = selected_model
    
    elif llm_provider == "Custom LLM":
        # Custom LLM configuration
        has_custom_config = bool(st.session_state.get('custom_api_key')) and bool(st.session_state.get('custom_base_url'))
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if has_custom_config:
                st.success("✅ Custom LLM configured")
                st.info(f"Base URL: {st.session_state.get('custom_base_url')}")
            else:
                # Base URL input
                base_url = st.text_input(
                    "API Base URL",
                    value=st.session_state.get('custom_base_url', ''),
                    key="popup_custom_base_url_input",
                    help="Enter the base URL for your custom LLM API (e.g., http://localhost:11434/v1)",
                    placeholder="http://localhost:11434/v1"
                )
                
                # API Key input
                col1_inner, col2_inner = st.columns([2, 1], vertical_alignment='bottom')
                with col1_inner:
                    api_key = st.text_input(
                        "API Key",
                        type="password",
                        value=st.session_state.get('custom_api_key', ''),
                        key="popup_custom_api_key_input",
                        help="Enter your custom LLM API key (use 'ollama' for Ollama)",
                        placeholder="your-api-key or 'ollama'"
                    )
                with col2_inner:
                    custom_confirm = st.button('Confirm', key='popup_custom_api_key_confirm_btn')
                
                if base_url and api_key and custom_confirm:
                    st.session_state.custom_base_url = base_url
                    st.session_state.custom_api_key = api_key
                    st.success("✅ Custom LLM configured!")
                    test_connection()
        
        with col2:
            if has_custom_config:
                if st.button("🗑️ Clear", key="clear_custom_key_btn", help="Clear Custom LLM Config"):
                    if 'custom_api_key' in st.session_state:
                        del st.session_state.custom_api_key
                    if 'custom_base_url' in st.session_state:
                        del st.session_state.custom_base_url
                    if 'custom_model_name' in st.session_state:
                        del st.session_state.custom_model_name
                    st.session_state.agent = None  # Reset agent
                    st.success("Custom LLM config cleared!")
                    st.rerun()
        
        # Model selection for Custom LLM
        if has_custom_config or not has_custom_config:  # Always show model selection
            custom_models = {
                "Azure OpenAI": [
                    "cts-vibecode-gpt-4.1",
                    "gpt-4",
                    "gpt-4-turbo",
                    "gpt-35-turbo"
                ],
                "Ollama Models": [
                    "llama2",
                    "llama2:13b",
                    "llama2:70b",
                    "codellama",
                    "codellama:13b",
                    "mistral",
                    "mistral:7b",
                    "mixtral:8x7b",
                    "phi3",
                    "qwen2",
                    "gemma"
                ],
                "Local/Custom": [
                    "gpt-3.5-turbo",
                    "gpt-4",
                    "claude-3-sonnet",
                    "custom-model"
                ]
            }
            
            # Flatten the model options
            all_custom_models = []
            for category, models in custom_models.items():
                all_custom_models.extend(models)
            
            current_model = st.session_state.get('custom_model_name', 'cts-vibecode-gpt-4.1')
            
            # Model selection dropdown
            selected_model = st.selectbox(
                "Model Name",
                all_custom_models,
                index=all_custom_models.index(current_model) if current_model in all_custom_models else 0,
                key="popup_custom_model_selectbox",
                help="Choose or select a model name"
            )
            
            # Optional custom model input for models not in the list
            if selected_model == "custom-model":
                custom_input = st.text_input(
                    "Enter Custom Model Name",
                    value="",
                    key="popup_custom_model_input",
                    help="Enter the exact model name",
                    placeholder="e.g., my-custom-model"
                )
                if custom_input:
                    selected_model = custom_input
            
            if selected_model:
                st.session_state.custom_model_name = selected_model
                st.session_state.model_name = selected_model
    
    # Initialize agent if we have everything needed
    if st.session_state.get('db_connected') and not st.session_state.get('agent'):
        llm_provider = st.session_state.get('llm_provider', 'openai')
        api_key_available = False
        
        if llm_provider == 'openai' and st.session_state.get('openai_api_key'):
            api_key_available = True
        elif llm_provider == 'gemini' and st.session_state.get('gemini_api_key'):
            api_key_available = True
        elif llm_provider == 'custom' and st.session_state.get('custom_api_key') and st.session_state.get('custom_base_url'):
            api_key_available = True
        
        if api_key_available:
            try:
                connection_string = st.session_state.get('connection_string')
                st.session_state.agent = ClinicalTrialsAgent(
                    connection_string, 
                    st.session_state.schema_processor,
                    llm_provider=st.session_state.llm_provider,
                    model_name=st.session_state.model_name
                )
            except Exception as e:
                st.error(f"Failed to initialize agent: {str(e)}")


def render_configuration_summary():
    """Render a summary of current configuration with clear buttons"""
    st.markdown("#### ⚙️ Current Configuration")
    
    # Show current provider and model
    llm_provider = st.session_state.get('llm_provider', 'openai')
    model_name = st.session_state.get('model_name', 'gpt-4')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write(f"**Provider:** {llm_provider.upper()}")
        st.write(f"**Model:** {model_name}")
        if llm_provider == 'openai':
            st.write(f"**API Key:** {st.session_state.get('openai_api_key')}")
        elif llm_provider == 'gemini':
            st.write(f"**API Key:** {st.session_state.get('gemini_api_key')}")
        elif llm_provider == 'custom':
            st.write(f"**API Key:** {st.session_state.get('custom_api_key')}")
            st.write(f"**Base URL:** {st.session_state.get('custom_base_url')}")
        
        # Show API key status
        if llm_provider == 'openai':
            if st.session_state.get('openai_api_key'):
                st.success("✅ OpenAI API Key configured")
            else:
                st.error("❌ OpenAI API Key missing")
        elif llm_provider == 'gemini':
            if st.session_state.get('gemini_api_key'):
                st.success("✅ Gemini API Key configured")
            else:
                st.error("❌ Gemini API Key missing")
        elif llm_provider == 'custom':
            if st.session_state.get('custom_api_key'):
                st.success("✅ Custom LLM configured")
            else:
                st.error("❌ Custom LLM API Key missing")
    
    with col2:
        # Clear API key button
        if llm_provider == 'openai' and st.session_state.get('openai_api_key'):
            if st.button("🗑️ Clear OpenAI Key", key="summary_clear_openai_btn"):
                if 'openai_api_key' in st.session_state:
                    del st.session_state.openai_api_key
                if 'OPENAI_API_KEY' in os.environ:
                    del os.environ['OPENAI_API_KEY']
                st.session_state.agent = None
                st.success("OpenAI API Key cleared!")
                st.rerun()
        elif llm_provider == 'gemini' and st.session_state.get('gemini_api_key'):
            if st.button("🗑️ Clear Gemini Key", key="summary_clear_gemini_btn"):
                if 'gemini_api_key' in st.session_state:
                    del st.session_state.gemini_api_key
                if 'GEMINI_API_KEY' in os.environ:
                    del os.environ['GEMINI_API_KEY']
                st.session_state.agent = None
                st.success("Gemini API Key cleared!")
                st.rerun()
        elif llm_provider == 'custom' and st.session_state.get('custom_api_key'):
            if st.button("🗑️ Clear Custom Key", key="summary_clear_custom_btn"):
                if 'custom_api_key' in st.session_state:
                    del st.session_state.custom_api_key
                if 'custom_base_url' in st.session_state:
                    del st.session_state.custom_base_url
                if 'custom_model_name' in st.session_state:
                    del st.session_state.custom_model_name
                st.session_state.agent = None
                st.success("Custom LLM config cleared!")
                st.rerun()
        
        # Reconfigure button
        if st.button("🔧 Reconfigure", key="reconfigure_btn"):
            # Open configuration dialog
            st.session_state.show_config = True
            st.rerun()

def render_floating_chat_button():
    """Render floating chat button using streamlit_float"""
    
    # Create the floating chat button container
    button_container = st.container()
    
    with button_container:
        # Chat toggle button with unique key based on session state
        button_key = f"floating_chat_toggle_btn_{id(st.session_state)}"
        if st.button("💬", key=button_key, help="Open Clinical Trials Assistant"):
            st.session_state.chat_widget_open = not st.session_state.chat_widget_open
            st.rerun()
    
    # Float the button to bottom-right corner
    button_container.float("bottom: 20px; right: 20px; width: 60px; height: 60px; border-radius: 50%; background: linear-gradient(135deg, #1f77b4, #4dabf7); box-shadow: 0 4px 20px rgba(31, 119, 180, 0.3); border: none; z-index: 1000;")


def render_floating_chat_popup():
    """Render the floating chat popup window using streamlit_float"""
    
    # Create chat popup container
    chat_container = st.container()
    
    with chat_container:
        # Create the main container structure
        st.markdown('<div class="chat-popup-container">', unsafe_allow_html=True)
        
        # Fixed Header Section
        st.markdown('<div class="chat-header">', unsafe_allow_html=True)
        st.markdown('<div class="chat-header-buttons">', unsafe_allow_html=True)
        header_col1, header_col2, header_col3 = st.columns([3, 1, 1])
        
        with header_col1:
            st.markdown("##### 🔬 Clinical Trials Assistant")
        
        with header_col2:
            # Initialize confirmation state if not exists
            if 'confirm_clear_chat' not in st.session_state:
                st.session_state.confirm_clear_chat = False
            
            if not st.session_state.confirm_clear_chat:
                if st.button("🗑️", key="popup_new_chat_btn", help="New Chat - Clear History"):
                    if st.session_state.messages:  # Only ask for confirmation if there are messages
                        st.session_state.confirm_clear_chat = True
                        st.rerun()
                    else:
                        st.info("No messages to clear!")
            else:
                # Show confirmation buttons
                col_confirm_1, col_confirm_2 = st.columns(2)
                with col_confirm_1:
                    if st.button("✅", key="confirm_clear_yes", help="Yes, clear chat"):
                        # Clear all chat messages
                        st.session_state.messages = []
                        # Clear agent memory if available
                        if st.session_state.agent and hasattr(st.session_state.agent, 'memory'):
                            st.session_state.agent.memory.clear()
                        st.session_state.confirm_clear_chat = False
                        st.success("Chat cleared! Starting fresh conversation.")
                        st.rerun()
                with col_confirm_2:
                    if st.button("❌", key="confirm_clear_no", help="Cancel"):
                        st.session_state.confirm_clear_chat = False
                        st.rerun()
        
        with header_col3:
            if st.button("✕", key="popup_close_btn", help="Close Chat"):
                st.session_state.chat_widget_open = False
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)  # close chat-header-buttons
        st.markdown('</div>', unsafe_allow_html=True)  # close chat-header
        
        # Scrollable Content Area
        st.markdown('<div class="chat-content">', unsafe_allow_html=True)
        
        # Configuration status check
        llm_provider = st.session_state.get('llm_provider', 'openai')
        has_api_key = False
        
        if llm_provider == 'openai':
            has_api_key = bool(st.session_state.get('openai_api_key'))
        elif llm_provider == 'gemini':
            has_api_key = bool(st.session_state.get('gemini_api_key'))
        elif llm_provider == 'custom':
            has_api_key = bool(st.session_state.get('custom_api_key')) and bool(st.session_state.get('custom_base_url'))
        
        # Show configuration if needed or requested
        if not has_api_key or st.session_state.get('show_config', False):
            render_llm_configuration()
            if st.session_state.get('show_config', False):
                # Add close configuration button
                if st.button("✅ Done", key="config_done_btn"):
                    st.session_state.show_config = False
                    st.rerun()
            st.markdown("---")
            st.markdown('</div>', unsafe_allow_html=True)  # close chat-content
            st.markdown('</div>', unsafe_allow_html=True)  # close chat-popup-container
            return
        
        # Show database status
        if not st.session_state.get('database_loaded'):
            st.info("ℹ️ Please test database connection in the sidebar first.")
            st.markdown('</div>', unsafe_allow_html=True)  # close chat-content
            st.markdown('</div>', unsafe_allow_html=True)  # close chat-popup-container
            return
        
        # Configuration summary
        with st.expander("⚙️ Current Configuration", expanded=False):
            render_configuration_summary()
        
        # Show confirmation message or welcome message
        if st.session_state.get('confirm_clear_chat', False):
            st.warning("⚠️ **Clear Chat History?** This will delete all previous messages. Use the ✅ or ❌ buttons above to confirm.")
        elif not st.session_state.messages:
            st.markdown("""
            **👋 Welcome! Ask me about clinical trials:**
            
            *Examples:*
            - "Phase 3 cancer trials in 2023"
            - "Top 10 sponsors by studies"
            - "Most common conditions"
            """)
        
        # Chat messages container
        messages_container = st.container()
        with messages_container:
            # Display conversation history (limit to last 10 messages for popup)
            recent_messages = st.session_state.messages[-10:] if len(st.session_state.messages) > 10 else st.session_state.messages
            for message in recent_messages:
                # Skip showing loading messages in history, but show current loading message
                if message.get("is_loading") and message != st.session_state.messages[-1]:
                    continue
                
                # Custom message styling based on role
                if message["role"] == "user":
                    # User message - left aligned
                    avatar_html = '<div class="message-avatar user-avatar">👤</div>'
                    message_html = f'''
                    <div class="user-message">
                        {avatar_html}
                        <div class="message-content">
                            {message["content"]}
                        </div>
                    </div>
                    '''
                    st.markdown(message_html, unsafe_allow_html=True)
                    
                else:  # assistant message
                    # Assistant message - right aligned
                    avatar_html = '<div class="message-avatar assistant-avatar">🤖</div>'
                    loading_class = " loading-message" if message.get("is_loading") else ""
                    
                    message_html = f'''
                    <div class="assistant-message">
                        <div class="message-content{loading_class}">
                            {message["content"]}
                        </div>
                        {avatar_html}
                    </div>
                    '''
                    st.markdown(message_html, unsafe_allow_html=True)
                    
                    # Show SQL queries if available (aligned to the right like assistant messages)
                    if message.get("sql_queries") and not message.get("is_loading"):
                        # Create a container that aligns to the right
                        col1, col2 = st.columns([1, 4])  # Empty column on left, content on right
                        with col2:
                            with st.expander("📊 View SQL Queries", expanded=False):
                                for i, query in enumerate(message["sql_queries"], 1):
                                    st.markdown(f"**Query {i}:**")
                                    st.code(query, language="sql")
        
        st.markdown('</div>', unsafe_allow_html=True)  # close chat-content
        
        # Fixed Chat Input Area
        st.markdown('<div class="chat-input-area">', unsafe_allow_html=True)
        
        # Chat input
        if prompt := st.chat_input("Ask about clinical trials...", key="popup_chat_input"):
            # Add user message
            user_message = {"role": "user", "content": prompt}
            st.session_state.messages.append(user_message)
            
            # Add a temporary loading message
            loading_message = {
                "role": "assistant", 
                "content": "🤖 Analyzing your query...",
                "is_loading": True
            }
            st.session_state.messages.append(loading_message)
            st.rerun()  # Refresh to show the loading message
        
        # Process any pending queries (this runs after rerun)
        if st.session_state.messages and st.session_state.messages[-1].get("is_loading"):
            # Remove the loading message
            loading_msg = st.session_state.messages.pop()
            user_prompt = st.session_state.messages[-1]["content"]  # Get the user's prompt
            
            # Generate response
            if st.session_state.agent:
                result = st.session_state.agent.query(user_prompt)
                
                if result.get("success"):
                    response_message = {
                        "role": "assistant",
                        "content": result["response"],
                        "success": True,
                        "timestamp": result["timestamp"],
                        "sql_queries": result.get("sql_queries", [])
                    }
                else:
                    response_message = {
                        "role": "assistant",
                        "content": f"Error: {result.get('error', 'Unknown error')}",
                        "success": False,
                        "timestamp": result["timestamp"]
                    }
                
                st.session_state.messages.append(response_message)
                st.rerun()
            else:
                error_message = {
                    "role": "assistant",
                    "content": "❌ Agent not initialized. Please check your configuration.",
                    "success": False
                }
                st.session_state.messages.append(error_message)
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)  # close chat-input-area
        st.markdown('</div>', unsafe_allow_html=True)  # close chat-popup-container
    
    # Float the chat popup
    chat_container.float("""
        position: fixed;
        bottom: 90px;
        right: 20px;
        width: 450px;
        height: 600px;
        background: white;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        border: 1px solid #e0e0e0;
        z-index: 999;
        padding: 15px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
    """)

def render_chat_boat():
    """Main application function"""
    # Apply chat-specific CSS
    st.markdown("""
    <style>
        /* Custom Chat Message Styling */
        .user-message {
            display: flex;
            justify-content: flex-start;
            margin: 5px 0;
        }
        
        .user-message .message-content {
            background-color: #e3f2fd;
            color: #0d47a1;
            padding: 6px 8px;
            border-radius: 18px 18px 18px 4px;
            max-width: 80%;
            margin-left: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            word-wrap: break-word;
            white-space: pre-wrap;
        }
        
        .assistant-message {
            display: flex;
            justify-content: flex-end;
            margin: 10px 0;
        }
        
        .assistant-message .message-content {
            background-color: #1f77b4;
            color: white;
            padding: 6px 8px;
            border-radius: 18px 18px 4px 18px;
            max-width: 80%;
            margin-right: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            word-wrap: break-word;
            white-space: pre-wrap;
        }
        
        .message-avatar {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            font-weight: bold;
            flex-shrink: 0;
        }
        
        .user-avatar {
            background-color: #2196f3;
            color: white;
        }
        
        .assistant-avatar {
            background-color: #4caf50;
            color: white;
        }
        
        .loading-message {
            background-color: #fff3e0 !important;
            color: #e65100 !important;
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }
        
        /* Hide default streamlit chat styling */
        .stChatMessage {
            display: none !important;
        }
        
        /* Chat popup header styling */
        .chat-header-buttons button {
            border-radius: 50%;
            width: 35px;
            height: 35px;
            border: none;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .chat-header-buttons button:hover {
            transform: scale(1.1);
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }
        
        /* Fixed header layout */
        .chat-popup-container {
            display: flex;
            flex-direction: column;
            height: 100%;
            max-height: 600px;
        }
        
        .chat-header {
            position: sticky;
            top: 0;
            background: white;
            border-bottom: 1px solid #e0e0e0;
            padding: 10px 0;
            z-index: 10;
            flex-shrink: 0;
        }
        
        .chat-content {
            flex: 1;
            overflow-y: auto;
            padding: 10px 0;
            min-height: 0;
        }
        
        .chat-input-area {
            position: sticky;
            bottom: 0;
            background: white;
            border-top: 1px solid #e0e0e0;
            padding: 10px 0 0 0;
            z-index: 10;
            flex-shrink: 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize streamlit_float
    float_init()
    
    # Setup sidebar (always visible)
    # setup_sidebar()
    
    # Optionally render the landing page
    # render_landing_page()
    
    # Always render the floating chat button
    render_floating_chat_button()
    
    # Show chat popup if open
    if st.session_state.chat_widget_open:
        render_floating_chat_popup()


# render_chat_boat() 