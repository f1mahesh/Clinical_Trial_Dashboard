# 🔬 Clinical Trial Analytics Dashboard

A modern, modular Streamlit application for comprehensive clinical trial data analysis using the AACT (Aggregate Analysis of ClinicalTrials.gov) database.

## 🚀 Features

### 📊 Core Analytics Modules
- **Home Dashboard**: Central navigation hub with database setup and overview
- **Trial Landscape Overview**: Comprehensive analysis of trial distribution, phases, and geographic patterns
- **Success & Completion Analytics**: Completion rates, results reporting, and success factor analysis
- **Innovation & Market Insights**: Intervention trends and emerging therapeutic areas
- **Access & Equity Analysis**: Geographic and demographic inclusion analysis
- **Investigator Ranking**: ML-powered investigator performance analysis
- **Facility Ranking**: Facility performance benchmarking and analysis
- **Study Reports**: Detailed individual study analysis and profiles
- **Sponsor Profiles**: Sponsor performance and portfolio analysis
- **Predictive Analytics**: Advanced ML models for trial success prediction
- **Trail Feasibility & Site Selection**: Advanced location optimization using population demographics
- **Executive Summary**: High-level insights and strategic recommendations

### 🛠️ Technical Features
- **Modular Architecture**: Clean, maintainable code structure with separation of concerns
- **Externalized Configuration**: SQL queries stored in YAML configuration files
- **Modern UI**: Professional, responsive design with custom styling and fragments
- **Database Flexibility**: Support for sample data, local PostgreSQL, and real AACT databases
- **Advanced Caching**: TTL-based caching with configurable expiration (24-hour default)
- **Pagination Support**: Efficient data display with customizable pagination (1000 records per page)
- **External API Integration**: Real-time data from US Census Bureau and World Bank APIs
- **Population Demographics**: Age/gender demographic matching for site selection
- **Robust Error Handling**: Comprehensive error handling and user feedback
- **Logging System**: Structured logging with module-specific log files
- **Connection Management**: Intelligent database connection pooling and management
- **Data Visualization**: Interactive charts with Plotly and custom styling

## 📁 Project Structure

```
Clinical_trail_Dashboard/
├── app.py                                  # Main application entry point
├── requirements.txt                        # Python dependencies
├── README.md                              # Project documentation
├── Logs/                                  # Application log files
│   └── trail_feasibility.log             # Trail feasibility module logs
├── config/
│   └── queries.yaml                       # Externalized SQL queries
├── src/
│   ├── database/
│   │   ├── __init__.py
│   │   └── connection_manager.py          # Database connection management
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config_loader.py               # Configuration management
│   │   ├── styling.py                     # Custom UI styling
│   │   ├── common_service.py              # Shared utility functions
│   │   └── pharma_gen.py                  # Pharmaceutical data generators
│   └── Module/
│       ├── __init__.py
│       ├── home.py                        # Landing page and navigation hub
│       ├── trial_landscape.py             # Trial landscape analysis
│       ├── success_analytics.py           # Success metrics analysis
│       ├── innovation_insights.py         # Innovation analysis
│       ├── access_equity.py               # Access and equity analysis
│       ├── investigator_ranking.py        # Investigator analysis
│       ├── facility_ranking.py            # Facility analysis
│       ├── study_reports.py               # Study reports
│       ├── sponsor_profiles.py            # Sponsor analysis
│       ├── predictive_analytics.py        # Predictive models
│       ├── trail_feasibility.py           # Site selection and feasibility analysis
│       └── executive_summary.py           # Executive dashboard
├── pages_bkp/                             # Backup of original page-based structure
│   └── [10 original page files]          # Historical page implementations
└── clinical_trial_analytics_dashboard.py  # Original monolithic file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Clinical_trail_Dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Access the dashboard**
   - Open your browser and navigate to `http://localhost:8501`
   - The application will start with the Home page

## 🗄️ Database Setup

The application supports three database connection types with intelligent connection management:

### Option 1: Sample Database (Recommended for Testing)
- Click "🚀 Load Sample Database" on the Home page
- Instantly creates a synthetic SQLite database with 1,000 sample trials
- No credentials required - perfect for exploring dashboard capabilities
- Includes realistic trial distributions across sponsors, phases, and conditions
- Enhanced sample data with interventions, investigators, and calculated values

### Option 2: Local PostgreSQL Database (For Organizations)
- Connect to your organization's local PostgreSQL instance
- Supports custom database schemas and local AACT copies
- Enhanced connection pooling and management
- Use for improved performance with local data copies
- Configure via the Home page database settings

### Option 3: Real AACT Database (For Production Use)
1. Register for free access at [aact.ctti-clinicaltrials.org](https://aact.ctti-clinicaltrials.org)
2. Obtain your database credentials
3. Use the Home page configuration to connect to the real AACT database
4. Access 400,000+ real clinical trials with live data
5. Monthly updates with latest trial registrations

### Database Features
- **Connection Testing**: Verify connectivity before running analyses
- **Connection Persistence**: Maintain connections across sessions
- **Intelligent Caching**: Cache database queries for improved performance
- **Error Recovery**: Robust error handling with user-friendly messages
- **Security**: Masked password fields and secure connection management

## 🎯 Usage Guide

### Navigation
- Use the tab navigation at the top to switch between analysis modules
- Each module provides focused insights on specific aspects of clinical trials
- Start with the Home page for an overview and database setup

### Key Modules

#### 📊 Trial Landscape
- Overview of trial distribution and trends
- Phase and status analysis
- Geographic insights and sponsor analysis
- Top conditions and therapeutic areas

#### ✅ Success Analytics
- Completion rates by phase and sponsor type
- Results reporting compliance analysis
- Duration analysis and efficiency metrics
- Success factor identification

#### 🏠 Home Page
- Database connection setup and testing
- Dashboard overview and navigation guide
- Getting started tips for different user types
- Quick statistics and feature overview

#### 🔬 Trail Feasibility & Site Selection
- **Population Demographics Integration**: Real-time data from US Census Bureau and World Bank APIs
- **Site Optimization**: Identify optimal locations based on historical trial data and regional demographics
- **Suitability Scoring**: Advanced algorithms to calculate recruitment potential scores
- **Geographic Analysis**: State-level and country-level demographic matching
- **Interactive Visualizations**: Population vs enrollment analysis, distribution charts
- **Exportable Reports**: Download comprehensive CSV reports for site selection decisions
- **Multiple Data Sources**: 
  - US States: Census Bureau American Community Survey (ACS) 5-year estimates
  - Countries: World Bank Population Statistics with age/gender breakdowns
- **Age/Gender Matching**: Precise demographic matching with proportional age group overlaps
- **Historical Analysis**: Leverage completed trial data for recruitment predictions

## 🔧 Configuration

### SQL Queries
All SQL queries are externalized in `config/queries.yaml`:
- Easy to modify and maintain
- No hardcoded queries in Python code
- Organized by analysis module
- Supports parameterized queries

### Customization
- Modify `src/utils/styling.py` for custom UI styling
- Update `config/queries.yaml` for custom database queries
- Add new analysis modules in `src/Module/`
- Extend database functionality in `src/database/`

## 🛠️ Development

### Adding New Analysis Modules
1. Create a new file in `src/Module/`
2. Implement the render function following the existing pattern
3. Add the module to the main navigation in `app.py`
4. Add corresponding queries to `config/queries.yaml`

### Code Quality
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Use type hints where appropriate
- Implement proper error handling

## 🔧 Advanced Features

### Performance Optimization
- **TTL Caching**: Configurable time-to-live caching (24-hour default)
- **Streamlit Fragments**: Efficient UI updates with `@st.fragment` decorators
- **Query Optimization**: Parameterized queries with connection pooling
- **Lazy Loading**: On-demand data loading for condition names and large datasets

### User Experience Enhancements
- **Pagination**: Handle large datasets with customizable pagination (1000 records per page)
- **Progressive Loading**: Step-by-step progress indicators for long-running analyses
- **Interactive Charts**: Hover data, zoom, and filtering capabilities
- **Responsive Design**: Modern UI that adapts to different screen sizes
- **Export Capabilities**: Download analysis results as CSV with proper formatting

### Data Processing
- **Demographic Matching**: Sophisticated age/gender matching algorithms
- **Suitability Scoring**: Normalized scoring using logarithmic scales (0-100 range)
- **Geographic Filtering**: Country and state-level filtering with API integration
- **Historical Analysis**: Leverage past trial data for future predictions

### Logging & Monitoring
- **Module-Specific Logs**: Separate log files for different analysis modules
- **Structured Logging**: Consistent log format with timestamps and levels
- **Error Tracking**: Comprehensive error logging for debugging
- **Performance Metrics**: Track query execution times and API response times

## 📊 Data Sources

### AACT Database
- **Source**: ClinicalTrials.gov via AACT (Aggregate Analysis of ClinicalTrials.gov)
- **Update Frequency**: Monthly
- **Data Types**: Clinical trial registrations, results, and metadata
- **Access**: Free registration required for real database access

### Sample Data
- **Source**: Synthetic data generated for demonstration
- **Coverage**: 1,000 sample trials with realistic distributions
- **Purpose**: Testing and exploration without database setup

### External APIs (Trail Feasibility Module)
- **US Census Bureau API**
  - **Source**: American Community Survey (ACS) 5-year estimates
  - **Data**: Detailed age/gender demographics by state
  - **Update Frequency**: Annual updates
  - **Coverage**: 18 detailed age groups for precise matching
  
- **World Bank API**
  - **Source**: World Bank Population Statistics
  - **Data**: Country-level population by age groups and gender
  - **Update Frequency**: Annual updates
  - **Coverage**: 15+ major countries with demographic breakdowns
  
- **Population Matching**
  - **Age Range Processing**: Intelligent parsing of clinical trial eligibility criteria
  - **Proportional Matching**: Accurate population estimates with age group overlaps
  - **Geographic Targeting**: State-level (US) and country-level analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support & Troubleshooting

### Common Issues

#### Database Connection Problems
- **Sample Database**: Ensure you have write permissions in the application directory
- **PostgreSQL**: Verify host, port, username, and password are correct
- **AACT Database**: Check your registration status at aact.ctti-clinicaltrials.org
- **Connection Test**: Use the "Test Connection" button before running analyses

#### Trail Feasibility Module Issues
- **API Timeouts**: External APIs may have rate limits; wait a moment and retry
- **No Results**: Try broader condition terms (e.g., "diabetes" instead of "type 2 diabetes")
- **Population Data Errors**: Some countries/states may not have complete demographic data

#### Performance Issues
- **Large Datasets**: Use pagination and filters to reduce data volume
- **Slow Loading**: Clear browser cache or restart the application
- **Memory Issues**: Try using the sample database for testing large analyses

### Getting Help
1. Check the documentation in this README
2. Review the code comments and docstrings
3. Check log files in the `Logs/` directory for detailed error messages
4. Open an issue on the repository with log details
5. Contact the development team

### Tips for Best Results
- **Start Small**: Begin with sample data before connecting to real databases
- **Use Filters**: Apply geographic and demographic filters to focus your analysis
- **Export Data**: Download CSV reports for offline analysis and sharing
- **Monitor Performance**: Check the browser console for any JavaScript errors
- **Regular Updates**: Keep dependencies updated for best performance

## 📦 Dependencies

### Core Requirements
```bash
streamlit>=1.28.0          # Web application framework
pandas>=1.5.0              # Data manipulation and analysis
plotly>=5.15.0             # Interactive visualizations
psycopg2-binary>=2.9.0     # PostgreSQL adapter
sqlalchemy>=1.4.0          # SQL toolkit and ORM
requests>=2.31.0           # HTTP library for API calls
pyyaml>=6.0                # YAML parser for configuration
```

### Optional Dependencies (for Trail Feasibility)
```bash
numpy>=1.24.0              # Mathematical functions
```

## 🔄 Migration from Original

### What Changed
- **Architecture**: Monolithic → Modular structure with proper separation of concerns
- **Navigation**: Sidebar → Tab-based navigation with modern UI
- **Configuration**: Hardcoded → Externalized queries in YAML files
- **Styling**: Inline CSS → Modular styling system with fragments
- **Database**: Single class → Comprehensive connection manager with pooling
- **Features**: Basic analytics → Advanced analytics with external API integration
- **Performance**: No caching → TTL-based caching with 24-hour expiration
- **UI**: Static → Dynamic with pagination and progressive loading
- **Data Sources**: Single database → Multiple sources (database + external APIs)

### Benefits
- **Maintainability**: Easier to modify and extend with clear module boundaries
- **Scalability**: Modular design supports growth and new feature additions
- **User Experience**: Modern, intuitive interface with responsive design
- **Performance**: Optimized caching, queries, and connection management
- **Code Quality**: Better organization, documentation, and error handling
- **Feature Rich**: Advanced analytics with population demographics integration
- **Flexibility**: Support for multiple database types and external data sources

## 🎉 Acknowledgments

- **AACT Team**: For providing the clinical trial database
- **Streamlit**: For the excellent web application framework
- **Plotly**: For interactive visualizations
- **ClinicalTrials.gov**: For the comprehensive trial registry 