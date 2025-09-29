"""
Trial Landscape Page
===================

Provides comprehensive overview of clinical trial distribution,
phases, status, and geographic analysis.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.utils.styling import create_insight_box
from src.utils.common_service import get_trial_landscape_metrics, check_database_connection
import pycountry_convert as pc
import geopandas as gpd

def render_trial_landscape_page():
    """Render the trial landscape overview page"""
    
    st.header("📊 Clinical Trial Landscape Overview",divider=True)
    
    # Check if database is connected
    if not check_database_connection():
        return
        
    # Load metrics
   
    metrics = get_trial_landscape_metrics()
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Studies", f"{metrics['total_studies']:,}")
    
    with col2:
        if not metrics['phase_distribution'].empty:
            most_common_phase = metrics['phase_distribution'].iloc[0]['phase']
            phase_count = metrics['phase_distribution'].iloc[0]['count']
            st.metric("Most Common Phase", most_common_phase, f"{phase_count} studies")
    
    with col3:
        if not metrics['status_distribution'].empty:
            completed_studies = metrics['status_distribution'][
                metrics['status_distribution']['overall_status'] == 'Completed'
            ]
            if not completed_studies.empty:
                completed_count = completed_studies.iloc[0]['count']
                completion_rate = round(100 * completed_count / metrics['total_studies'], 1)
                st.metric("Completion Rate", f"{completion_rate}%", f"{completed_count} completed")
    
    with col4:
        if not metrics['geographic_distribution'].empty:
            total_countries = len(metrics['geographic_distribution'])
            st.metric("Countries Represented", total_countries)
    
    # Visualizations
    st.subheader("📈 Landscape Visualizations")
    
    # Create visualization tabs
    viz_tab0, viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
        "📊 Study Registration Trend",
        "📊 Phase & Status", 
        "🌍 Geographic Distribution", 
        "🏢 Sponsor Analysis", 
        "📋 Top Conditions"
    ])
    
    with viz_tab0:
        render_study_trend_visualizations()
       
    with viz_tab1:
        render_phase_status_visualizations(metrics)
    
    with viz_tab2:
        render_geographic_visualizations(metrics)
    
    with viz_tab3:
        render_sponsor_visualizations(metrics)
    
    with viz_tab4:
        render_conditions_visualizations(metrics)
    
    # Data tables
    with st.expander("📋 Detailed Data Tables"):
        tab1, tab2, tab3, tab4 = st.tabs([
            "Phase Distribution", 
            "Status Distribution", 
            "Top Conditions", 
            "Geographic Distribution"
        ])
        
        with tab1:
            if not metrics['phase_distribution'].empty:
                st.dataframe(metrics['phase_distribution'], use_container_width=True)
            else:
                st.info("No phase distribution data available")
        
        with tab2:
            if not metrics['status_distribution'].empty:
                st.dataframe(metrics['status_distribution'], use_container_width=True)
            else:
                st.info("No status distribution data available")
        
        with tab3:
            if not metrics['top_conditions'].empty:
                st.dataframe(metrics['top_conditions'], use_container_width=True)
            else:
                st.info("No conditions data available")
        
        with tab4:
            if not metrics['geographic_distribution'].empty:
                st.dataframe(metrics['geographic_distribution'], use_container_width=True)
            else:
                st.info("No geographic data available")
    
    # Insights
    st.markdown("---")
    render_landscape_insights(metrics)

@st.fragment
def render_study_trend_visualizations():
    # Add time period selector
    c= st.columns([8,2])
    with c[1]:
        time_period = st.selectbox(
            "Select Time Period",
            ["Yearly", "Quarterly", "Monthly"],
            index=0
        )
    
    params = {'time_period':time_period}
    registration_data = st.session_state.db_manager.execute_query(
        st.session_state.config.get_query('trial_landscape', 'registration_query'),
        params
    )
    
    if not registration_data.empty:
        # Create trend visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=registration_data['period'],
            y=registration_data['count'],
            mode='lines+markers',
            name='Studies Registered'
        ))
        
        fig.update_layout(
            title=f"Study Registration Trend ({time_period})",
            xaxis_title="Time Period",
            yaxis_title="Number of Studies Registered",
            showlegend=True,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add summary statistics
        total_registrations = registration_data['count'].sum()
        avg_registrations = registration_data['count'].mean()
        
        st.markdown(f"""
        **Registration Summary:**
        - Total Registrations: {total_registrations:,.0f}
        - Average per {time_period.lower()}: {avg_registrations:,.1f}
        """)
    else:
        st.info("No registration trend data available")


def render_phase_status_visualizations(metrics):
    """Render phase and status visualizations"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not metrics['phase_distribution'].empty:
            fig_phase = px.pie(
                metrics['phase_distribution'], 
                values='count', 
                names='phase',
                title="Clinical Trial Phase Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_phase.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_phase, use_container_width=True)
        else:
            st.info("No phase distribution data available")
    
    with col2:
        if not metrics['status_distribution'].empty:
            fig_status = px.bar(
                metrics['status_distribution'].head(10),
                x='overall_status',
                y='count',
                title="Clinical Trial Status Distribution",
                color='count',
                color_continuous_scale='viridis'
            )
            fig_status.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_status, use_container_width=True)
        else:
            st.info("No status distribution data available")
    
    # Combined phase vs status heatmap
    if not metrics['phase_distribution'].empty and not metrics['status_distribution'].empty:
        st.subheader("Phase vs Status Analysis")
        
        # Create a simple cross-tabulation (this would need more complex query in real implementation)
        try:
            phase_status_query = """
            SELECT phase, overall_status, COUNT(*) as count
            FROM studies 
            WHERE phase IS NOT NULL
            GROUP BY phase, overall_status
            ORDER BY phase, overall_status
            """
            phase_status_data = st.session_state.db_manager.execute_query(phase_status_query)
            
            if not phase_status_data.empty:
                # Pivot the data for heatmap
                pivot_data = phase_status_data.pivot(index='phase', columns='overall_status', values='count').fillna(0)
                
                fig_heatmap = px.imshow(
                    pivot_data,
                    title="Phase vs Status Heatmap",
                    color_continuous_scale='viridis',
                    aspect='auto'
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
        except Exception as e:
            st.info("Phase vs Status heatmap not available for current data")


# --- Function to determine text color based on background color ---
def get_text_color_based_on_count(count_value, min_val, max_val, threshold=0.6):
    """
    Determines text color (white or black) based on normalized count value.
    Assumes a colorscale where higher values are darker (like 'YlGnBu', 'Viridis', 'Plasma').
    """
    if max_val == min_val: # Handle case where all counts are the same
        return "black" # Default to black if no variation

    # Normalize the count value to a 0-1 range
    normalized_count = (count_value - min_val) / (max_val - min_val)

    # If the normalized count is above the threshold, use white text (for darker backgrounds)
    # Otherwise, use black text (for lighter backgrounds)
    if normalized_count > threshold:
        return "white"
    else:
        return "black"

def plotByContinent(df,column_name):
    world = gpd.read_file("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson")
    # world
    # Dissolve country polygons into continent polygons
    # This groups all countries by their 'continent' attribute and merges their geometries
    continents_gdf = world.dissolve(by='CONTINENT')

    # Reset index to make 'continent' a regular column again for merging
    continents_gdf = continents_gdf.reset_index()
    

    # Rename the 'continent' column to match our data for merging
    # continents_gdf = continents_gdf.rename(columns={'continent': 'CONTINENT'})

    # Merge our count data with the GeoDataFrame of continents
    # This adds the 'Count' column to our geographical data
    merged_gdf = pd.merge(continents_gdf, df, on='CONTINENT', how='left')

    # Drop any continents that are not in our data (e.g., Antarctica if not provided)
    merged_gdf = merged_gdf.dropna(subset=[column_name])

    # Prepare centroids for placing text labels
    # Calculate the centroid (approximate center) of each continent's geometry
    merged_gdf['centroid'] = merged_gdf.geometry.centroid
    merged_gdf['lon'] = merged_gdf.centroid.x
    merged_gdf['lat'] = merged_gdf.centroid.y
    

    # Calculate min and max counts for normalization
    min_count = merged_gdf[column_name].min()
    max_count = merged_gdf[column_name].max()

    # Create a list of text colors for each continent
    text_colors = [
        get_text_color_based_on_count(row[column_name], min_count, max_count)
        for index, row in merged_gdf.iterrows()
    ]


    # --- 3. Map Generation using Plotly Graph Objects ---
    fig = go.Figure()

    # Add the choropleth trace for continent shapes
    fig.add_trace(
        go.Choropleth(
            geojson=merged_gdf.geometry.__geo_interface__, # GeoJSON data for continent shapes
            locations=merged_gdf.index, # Use the DataFrame index as locations
            z=merged_gdf[column_name],      # Data to color the continents by
            # 'Plasma', 'Cividis', 'Inferno', 'Magma', 'YlGnBu', 'RdPu', 'Greens', 'Blues'
            # colorscale='YlGnBu',        # Color scale for the map (changed from Viridis)
            colorscale = 'Blues',
            # autocolorscale=False,
            # reversescale=True,
            marker_line_color='darkgray',  # Continent border color
            marker_line_width=0.1,      # Continent border width
            colorbar_title=column_name,     # Title for the color bar
            hoverinfo='skip'            # Disable default hover info for choropleth
        )
    )

    # Add a scattergeo trace for text labels (continent name and count)
    fig.add_trace(
        go.Scattergeo(
            lon=merged_gdf['lon'],
            lat=merged_gdf['lat'],
            text=[f"{row['CONTINENT']}<br>{int(row[column_name]):,}" for index, row in merged_gdf.iterrows()],
            mode='text', # Display only text markers
            textfont=dict(
                color=text_colors, # Text color
                size=20,       # Text size
                weight="bold",
                family="Arial, sans-serif" # Font family
            ),
            hoverinfo='text', # Show text on hover
            showlegend=False # Do not show this trace in the legend
        )
    )

    # --- 4. Map Customization ---
    fig.update_layout(
        title_text='Clinical Trials by Continent', # Main title of the map
        # title_x=0.25, # Center the title
        geo=dict(
            showland=True,
            landcolor="lightgray", # Color of the land areas
            showocean=True,
            oceancolor="lightblue", # Color of the ocean areas
            showcountries=False,   # Hide country borders (only show continent shapes)
            showsubunits=False,    # Hide internal country subdivisions
            showcoastlines=False,  # Hide coastlines
            projection_type="natural earth", # Map projection
            lataxis_range=[-60, 90], # Adjust latitude range to focus on continents
            lonaxis_range=[-180, 180] # Adjust longitude range
        ),
        margin={"r":0,"t":50,"l":0,"b":0}, # Set margins
    )
    st.plotly_chart(fig)

def render_geographic_visualizations(metrics):
    """Render geographic distribution visualizations"""
    
    if not metrics['geographic_distribution'].empty:
        # Use pycountry_convert to map countries to continents
        import pycountry_convert as pc
        
        def country_to_continent(country_name):
            try:
                # Get country alpha-2 code
                country_alpha2 = pc.country_name_to_country_alpha2(country_name)
                # Get continent code
                continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
                # Get continent name
                continent_name = pc.convert_continent_code_to_continent_name(continent_code)
                return continent_name
            except:
                return "Unknown"
        
        # Add continent column
        continent_df = metrics['geographic_distribution'].copy()
        continent_df['CONTINENT'] = continent_df['country'].apply(country_to_continent)
        
        # Aggregate by continent
        continent_data = continent_df.groupby('CONTINENT')['studies_count'].sum().reset_index()
        
        plotByContinent(continent_data,'studies_count')
        
        # # Show continent breakdown table
        # st.markdown("### Continental Distribution")
        # st.dataframe(
        #     continent_data.sort_values('studies_count', ascending=False),
        #     hide_index=True
        # )
        
        # World map (simplified - would need country codes in real implementation)
        st.subheader("Geographic Distribution")
        
        # Create a simple map visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top 10 Countries by Trial Count:**")
            top_10 = metrics['geographic_distribution'].head(10)
            for idx, row in top_10.iterrows():
                st.markdown(f"• **{row['country']}**: {row['studies_count']} studies")
        
        with col2:
            st.markdown("**Geographic Diversity Metrics:**")
            total_countries = len(metrics['geographic_distribution'])
            total_studies = metrics['geographic_distribution']['studies_count'].sum()
            avg_per_country = total_studies / total_countries if total_countries > 0 else 0
            
            st.metric("Total Countries", total_countries)
            st.metric("Average Studies/Country", f"{avg_per_country:.1f}")
    else:
        st.info("No geographic distribution data available")

def render_sponsor_visualizations(metrics):
    """Render sponsor analysis visualizations"""
    
    if not metrics['sponsor_analysis'].empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Sponsor type distribution
            fig_sponsor = px.pie(
                metrics['sponsor_analysis'],
                values='studies_count',
                names='source_class',
                title="Clinical Trials by Sponsor Type",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_sponsor.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_sponsor, use_container_width=True)
        
        with col2:
            # Average enrollment by sponsor type
            fig_enrollment = px.bar(
                metrics['sponsor_analysis'],
                x='source_class',
                y='avg_enrollment',
                title="Average Enrollment by Sponsor Type",
                color='studies_count',
                color_continuous_scale='viridis'
            )
            fig_enrollment.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_enrollment, use_container_width=True)
        
        # Sponsor efficiency analysis
        st.subheader("Sponsor Efficiency Analysis")
        
        efficiency_data = metrics['sponsor_analysis'].copy()
        efficiency_data['efficiency_score'] = (
            efficiency_data['studies_count'] * efficiency_data['avg_enrollment']
        ) / efficiency_data['avg_duration'].fillna(1)
        
        fig_efficiency = px.scatter(
            efficiency_data,
            x='avg_enrollment',
            y='avg_duration',
            size='studies_count',
            color='source_class',
            title="Sponsor Efficiency: Enrollment vs Duration",
            hover_data=['studies_count', 'efficiency_score']
        )
        st.plotly_chart(fig_efficiency, use_container_width=True)
    else:
        st.info("No sponsor analysis data available")

def render_conditions_visualizations(metrics):
    """Render conditions analysis visualizations"""
    
    if not metrics['top_conditions'].empty:
        # Top conditions bar chart
        fig_conditions = px.bar(
            metrics['top_conditions'],
            x='name',
            y='studies_count',
            title="Top Clinical Trial Conditions",
            color='studies_count',
            color_continuous_scale='viridis'
        )
        fig_conditions.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_conditions, use_container_width=True)
        
        # Conditions word cloud (simplified representation)
        st.subheader("Condition Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top 10 Conditions:**")
            for idx, row in metrics['top_conditions'].head(10).iterrows():
                percentage = (row['studies_count'] / metrics['top_conditions']['studies_count'].sum()) * 100
                st.markdown(f"• **{row['name']}**: {row['studies_count']} studies ({percentage:.1f}%)")
        
        with col2:
            # Condition diversity metrics
            total_conditions = len(metrics['top_conditions'])
            total_studies = metrics['top_conditions']['studies_count'].sum()
            avg_per_condition = total_studies / total_conditions if total_conditions > 0 else 0
            
            st.metric("Total Conditions", total_conditions)
            st.metric("Average Studies/Condition", f"{avg_per_condition:.1f}")
            st.metric("Most Studied", metrics['top_conditions'].iloc[0]['name'])
    else:
        st.info("No conditions data available")

def render_landscape_insights(metrics):
    """Render landscape insights and analysis"""
    
    st.subheader("🔍 Key Insights")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        # Phase insights
        if not metrics['phase_distribution'].empty:
            most_common_phase = metrics['phase_distribution'].iloc[0]['phase']
            phase_count = metrics['phase_distribution'].iloc[0]['count']
            phase_percentage = (phase_count / metrics['total_studies']) * 100
            
            st.markdown(f"""
            **📊 Phase Distribution Insights:**
            - **Most Common Phase**: {most_common_phase} ({phase_percentage:.1f}% of all trials)
            - **Phase Diversity**: {len(metrics['phase_distribution'])} different phases represented
            - **Early vs Late Stage**: Analysis shows distribution across development stages
            """)
        
        # Status insights
        if not metrics['status_distribution'].empty:
            completed_studies = metrics['status_distribution'][
                metrics['status_distribution']['overall_status'] == 'Completed'
            ]
            if not completed_studies.empty:
                completion_rate = (completed_studies.iloc[0]['count'] / metrics['total_studies']) * 100
                st.markdown(f"""
                **✅ Completion Insights:**
                - **Completion Rate**: {completion_rate:.1f}% of trials are completed
                - **Active Trials**: {len(metrics['status_distribution'])} different status categories
                - **Success Tracking**: Comprehensive status monitoring available
                """)
    
    with insights_col2:
        # Geographic insights
        if not metrics['geographic_distribution'].empty:
            total_countries = len(metrics['geographic_distribution'])
            top_country = metrics['geographic_distribution'].iloc[0]['country']
            top_country_studies = metrics['geographic_distribution'].iloc[0]['studies_count']
            
            st.markdown(f"""
            **🌍 Geographic Insights:**
            - **Global Reach**: Trials conducted in {total_countries} countries
            - **Top Location**: {top_country} leads with {top_country_studies} studies
            - **Diversity**: Wide geographic distribution indicates global collaboration
            """)
        
        # Sponsor insights
        if not metrics['sponsor_analysis'].empty:
            top_sponsor = metrics['sponsor_analysis'].iloc[0]['source_class']
            top_sponsor_studies = metrics['sponsor_analysis'].iloc[0]['studies_count']
            
            st.markdown(f"""
            **🏢 Sponsor Insights:**
            - **Leading Sponsor Type**: {top_sponsor} with {top_sponsor_studies} studies
            - **Sponsor Diversity**: {len(metrics['sponsor_analysis'])} different sponsor types
            - **Collaboration**: Mix of industry, government, and academic sponsors
            """)
    
    # Recommendations
    st.markdown("---")
    st.subheader("💡 Strategic Recommendations")
    
    recommendations = [
        "**Focus on High-Impact Phases**: Prioritize Phase 2 and 3 trials for maximum clinical impact",
        "**Geographic Expansion**: Consider expanding trials to underserved regions for better diversity",
        "**Sponsor Collaboration**: Foster partnerships between different sponsor types for innovation",
        "**Condition Prioritization**: Align trial focus with high-prevalence conditions",
        "**Status Monitoring**: Implement robust tracking systems for trial progress and completion"
    ]
    
    for rec in recommendations:
        st.markdown(f"• {rec}") 


render_trial_landscape_page()