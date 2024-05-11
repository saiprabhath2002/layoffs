import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
import plotly.express as px

# Function to load and preprocess data
@st.cache_data
def load_data():
    data_path = 'layoffs_data(4).csv'
    data = pd.read_csv(data_path)
    data['Date'] = pd.to_datetime(data['Date'], errors = 'coerce')
    data['YearMonth'] = data['Date'].dt.to_period('M').dt.to_timestamp()
    data['Year'] = data['Date'].dt.year
    # data[data['Date'].notna()]
    return data

data = load_data()

# Styling for the app - Adding some CSS for better aesthetics
st.markdown("""
<style>
.main {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
}
.header {
    color: #0c4191;
    font-size: 24px;
    font-weight: bold;
}
.subheader {
    color: #34656d;
    font-size: 20px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar for settings and user control
st.sidebar.title('Settings')
# st.sidebar.info(
#     """
#     **Dark/Light Mode:** You can switch between dark and light mode by clicking on the hamburger menu 
#     in the upper right → Settings → Theme.
#     """
# )

st.sidebar.markdown('Instructor : Sougata  Mukherjea')
st.sidebar.markdown('**Course Code:** ELL824')
st.sidebar.markdown('**Group Members:**')
st.sidebar.markdown('Bogam Sai Prabhath (2023AIB2079)')
st.sidebar.markdown('Lakshay Kakkar (2023CSY7548)')

st.sidebar.header('Filter Options')
selected_years = st.sidebar.slider("Select the year range:", int(data['Date'].dt.year.min()), int(data['Date'].dt.year.max()), (int(data['Date'].dt.year.min()), int(data['Date'].dt.year.max())))
selected_countries = st.sidebar.multiselect('Select countries:', options=data['Country'].unique(), default=data['Country'].unique())

# Filtering data based on user selection
data_filtered = data[(data['Year'] >= selected_years[0]) & (data['Year'] <= selected_years[1]) & (data['Country'].isin(selected_countries))]

# Main dashboard title
st.title('Interactive Layoff Analysis Dashboard', anchor='main-title')

# Insert your plots here using Plotly or any other library
# Example: A placeholder plot with filtered data
# st.header('Sample Data Display')
# st.write(data_filtered.head())

# Add more plots and analysis as needed below
# Place to insert plots
# Use `st.plotly_chart`, `st.pyplot`, etc., to add plots here.

# Example Plot Insertion (Replace with actual plot calls)
# st.subheader('More Detailed Analysis Layoff Trends ', anchor='layoff-trends')
# st.plotly_chart(fig)  # Uncomment this line and replace 'fig' with your actual Plotly figure variable or similar for other plotting libraries.

# Placeholder for further visualizations
# st.subheader('More Detailed Analysis', anchor='detailed-analysis')
# st.write('Detailed charts and graphs can be placed here using the filtered data.')


# st.subheader('Average Percentage of Workforce Laid Off Over Time')
# percentage_over_time = data_filtered.groupby('YearMonth')['Percentage'].mean()
# fig = px.line(percentage_over_time, x=percentage_over_time.index, y='Percentage', labels={'index': 'Year-Month', 'Percentage': 'Average Percentage'})
# st.plotly_chart(fig)



st.subheader('Total Number of Layoffs by Country (Top 20)')
layoffs_by_country = data_filtered.groupby('Country')['Laid_Off_Count'].sum()
top_countries = layoffs_by_country.nlargest(20)
fig = px.bar(top_countries, x=top_countries.index, y='Laid_Off_Count', title="Top 20 Countries by Layoffs", labels={'index': 'Country', 'Laid_Off_Count': 'Total Layoffs'}, log_y=True)
st.plotly_chart(fig)



st.subheader('Correlation Between Funds Raised and Layoffs')
clean_data = data_filtered.dropna(subset=['Funds_Raised', 'Laid_Off_Count'])
fig = px.scatter(clean_data, x='Funds_Raised', y='Laid_Off_Count', trendline="ols", log_x=True, labels={'Funds_Raised': 'Funds Raised (in millions $)', 'Laid_Off_Count': 'Number of Layoffs'})
st.plotly_chart(fig)



st.subheader('Total Number of Layoffs by Company Stage')
layoffs_by_stage = data_filtered.groupby('Stage')['Laid_Off_Count'].sum().sort_values(ascending=False)
fig = px.bar(layoffs_by_stage, x=layoffs_by_stage.index, y='Laid_Off_Count', labels={'index': 'Company Stage', 'Laid_Off_Count': 'Total Layoffs'})
st.plotly_chart(fig)


st.subheader('Layoffs Over Time by Industry')
industry_time_series = data_filtered.groupby(['YearMonth', 'Industry'])['Laid_Off_Count'].sum().unstack(fill_value=0)
fig = px.line(industry_time_series, labels={'value': 'Number of Layoffs', 'variable': 'Industry'})
st.plotly_chart(fig)

import streamlit as st
import pandas as pd
import plotly.express as px

# Assuming data_filtered is already loaded and filtered accordingly

# Calculate total layoffs by industry
total_layoffs_by_industry = data_filtered.groupby('Industry')['Laid_Off_Count'].sum().sort_values()

# Get the top 5 industries
top_5_industries = total_layoffs_by_industry.nlargest(5).index.tolist()

# Get the bottom 5 industries
bottom_5_industries = total_layoffs_by_industry.nsmallest(5).index.tolist()

# Filtering data for top 5 industries
top_industries_data = data_filtered[data_filtered['Industry'].isin(top_5_industries)]
top_industry_time_series = top_industries_data.groupby(['YearMonth', 'Industry'])['Laid_Off_Count'].sum().unstack(fill_value=0)

# Filtering data for bottom 5 industries
bottom_industries_data = data_filtered[data_filtered['Industry'].isin(bottom_5_industries)]
bottom_industry_time_series = bottom_industries_data.groupby(['YearMonth', 'Industry'])['Laid_Off_Count'].sum().unstack(fill_value=0)

# Plotting for top 5 industries
st.subheader('Top 5 Industries by Layoffs Over Time')
fig_top = px.line(top_industry_time_series, labels={'value': 'Number of Layoffs', 'variable': 'Industry'}, title='Top 5 Industries by Layoffs Over Time')
st.plotly_chart(fig_top)

# Plotting for bottom 5 industries
st.subheader('Bottom 5 Industries by Layoffs Over Time')
fig_bottom = px.line(bottom_industry_time_series, labels={'value': 'Number of Layoffs', 'variable': 'Industry'}, title='Bottom 5 Industries by Layoffs Over Time')
st.plotly_chart(fig_bottom)


# st.subheader('Layoffs Over Time by Industry')
# industry_time_series = data_filtered.groupby(['YearMonth', 'Industry'])['Laid_Off_Count'].sum().unstack(fill_value=0)

# # Create the figure
# fig = go.Figure()

# # Add traces for each industry
# for industry, layoffs in industry_time_series.items():
#     fig.add_trace(go.Scatter(
#         x=layoffs.index, 
#         y=layoffs,
#         name=industry,
#         line=dict(width=2),  # Normal width
#         opacity=0.1,  # Make lines very dim by default
#         hoverinfo='name+y',  # Show industry name and layoffs on hover
#         mode='lines',
#         hovertemplate='<b>%{name}</b><br>%{y}<extra></extra>',  # Custom hover template for clarity
#     ))

# # Adjust layout to improve hover effect
# fig.update_traces(
#     mode='lines',
#     hoveron='points+fills',  # Allow hover effects on both points and line fills
#     line=dict(color='rgba(30, 30, 30, 0.15)'),  # Default dim color
#     hoverlabel=dict(bgcolor='white', font_size=14, font_family="Arial"),
#     selected=dict(line_color='blue', line_width=3.5),  # Highlight style
#     unselected=dict(line_color='rgba(30, 30, 30, 0.15)', line_width=2)  # Non-highlight style
# )

# fig.update_layout(
#     hovermode='closest',
#     title='Layoffs Over Time by Industry',
#     xaxis_title='Date',
#     yaxis_title='Number of Layoffs',
#     legend_title='Industry',
#     plot_bgcolor='white'
# )

# # Hover template adjustments
# fig.update_layout(
#     hoverlabel_align='right',
# )

# st.plotly_chart(fig, use_container_width=True)

# Filtering data based on user selection in sidebar
selected_years = st.sidebar.slider("Select the year range:", int(data['Date'].dt.year.min()), int(data['Date'].dt.year.max()), (int(data['Date'].dt.year.min()), int(data['Date'].dt.year.max())), key = 'year_range_slider')
selected_countries = st.sidebar.multiselect('Select countries:', options=data['Country'].unique(), default=data['Country'].unique(), key = 'new')
data_filtered = data[(data['Year'] >= selected_years[0]) & (data['Year'] <= selected_years[1]) & (data['Country'].isin(selected_countries))]

# Heatmap of Layoffs by Month and Industry
industry_time_series = data_filtered.groupby(['YearMonth', 'Industry'])['Laid_Off_Count'].sum().unstack(fill_value=0)
fig_heatmap = px.imshow(industry_time_series, labels=dict(x="Month", y="Industry", color="Number of Layoffs"),
                title="Heatmap of Layoffs by Month and Industry")
st.plotly_chart(fig_heatmap)

# Histogram of Funds Raised
fig_histogram = px.histogram(data_filtered, x='Funds_Raised', color='Industry',
                   title="Histogram of Funds Raised by Industry")
st.plotly_chart(fig_histogram)

# Box Plot of Layoffs by Company Stage
fig_box = px.box(data_filtered, x='Stage', y='Laid_Off_Count', title="Box Plot of Layoffs by Company Stage")
st.plotly_chart(fig_box)

# Layoffs Trend Line with Moving Average
data_filtered['MA_Layoffs'] = data_filtered['Laid_Off_Count'].rolling(window=12, min_periods=1).mean()
fig_trend = px.line(data_filtered, x='YearMonth', y='MA_Layoffs', title="Layoffs Trend Line with Moving Average")
st.plotly_chart(fig_trend)

# Proportional Area Chart of Layoffs by Country
country_layoffs = data_filtered.groupby('Country')['Laid_Off_Count'].sum().reset_index()
fig_area = px.scatter(country_layoffs, x='Country', y='Laid_Off_Count', size='Laid_Off_Count', title="Proportional Area Chart of Layoffs by Country",
                size_max=60)
st.plotly_chart(fig_area)

# st.subheader('Sector-specific Layoffs Over Time')

# # Define the industry categories
# tech_industries = ["AI", "Aerospace", "Crypto", "Data"]
# service_industries = ["Consumer", "Education", "Finance", "Fitness", "Food", "HR", "Healthcare", "Legal", "Marketing", "Media", "Real Estate", "Recruiting", "Retail", "Sales", "Security", "Support", "Transportation", "Travel"]
# manufacturing_industries = ["Construction", "Hardware", "Infrastructure", "Manufacturing"]

# # Filter data for each category
# tech_data = data_filtered[data_filtered['Industry'].isin(tech_industries)]
# service_data = data_filtered[data_filtered['Industry'].isin(service_industries)]
# manufacturing_data = data_filtered[data_filtered['Industry'].isin(manufacturing_industries)]

# # Plotting each category
# fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

# tech_layoffs = tech_data.groupby('YearMonth')['Laid_Off_Count'].sum()
# service_layoffs = service_data.groupby('YearMonth')['Laid_Off_Count'].sum()
# manufacturing_layoffs = manufacturing_data.groupby('YearMonth')['Laid_Off_Count'].sum()

# tech_layoffs.plot(ax=axs[0], title='Technology Sector Layoffs Over Time', marker='o', linestyle='-')
# service_layoffs.plot(ax=axs[1], title='Service Sector Layoffs Over Time', marker='o', linestyle='-')
# manufacturing_layoffs.plot(ax=axs[2], title='Manufacturing Sector Layoffs Over Time', marker='o', linestyle='-')

# for ax in axs:
#     ax.set_xlabel('Year-Month')
#     ax.set_ylabel('Number of Layoffs')
#     ax.xaxis.set_major_locator(mdates.YearLocator())
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#     plt.xticks(rotation=45)
#     ax.legend(title='Industry', bbox_to_anchor=(1.05, 1), loc='upper left')

# fig.tight_layout(pad=3.0)
# st.pyplot(fig)

tech_industries = ["AI", "Aerospace", "Crypto", "Data"]
service_industries = ["Consumer", "Education", "Finance", "Fitness", "Food", "HR", "Healthcare", "Legal", "Marketing", "Media", "Real Estate", "Recruiting", "Retail", "Sales", "Security", "Support", "Transportation", "Travel"]
manufacturing_industries = ["Construction", "Hardware", "Infrastructure", "Manufacturing"]

# Filter data for each category
tech_data = data_filtered[data_filtered['Industry'].isin(tech_industries)]
service_data = data_filtered[data_filtered['Industry'].isin(service_industries)]
manufacturing_data = data_filtered[data_filtered['Industry'].isin(manufacturing_industries)]

# Plotting each category using Plotly
# Tech Industries
st.subheader('Tech Sector Layoffs Over Time')
tech_layoffs = tech_data.groupby('YearMonth')['Laid_Off_Count'].sum().reset_index()
fig_tech = px.line(tech_layoffs, x='YearMonth', y='Laid_Off_Count', title='Technology Sector Layoffs')
st.plotly_chart(fig_tech)

# Service Industries
st.subheader('Service Sector Layoffs Over Time')
service_layoffs = service_data.groupby('YearMonth')['Laid_Off_Count'].sum().reset_index()
fig_service = px.line(service_layoffs, x='YearMonth', y='Laid_Off_Count', title='Service Sector Layoffs')
st.plotly_chart(fig_service)

# Manufacturing Industries
st.subheader('Manufacturing Sector Layoffs Over Time')
manufacturing_layoffs = manufacturing_data.groupby('YearMonth')['Laid_Off_Count'].sum().reset_index()
fig_manufacturing = px.line(manufacturing_layoffs, x='YearMonth', y='Laid_Off_Count', title='Manufacturing Sector Layoffs')
st.plotly_chart(fig_manufacturing)



st.subheader('Relationship Between Layoff Count and Percentage of Workforce Laid Off')
clean_data = data_filtered.dropna(subset=['Laid_Off_Count', 'Percentage'])
fig = px.scatter(clean_data, x='Laid_Off_Count', y='Percentage', log_x=True, labels={'Laid_Off_Count': 'Number of Layoffs', 'Percentage': 'Percentage of Workforce Laid Off'})
st.plotly_chart(fig)



st.subheader('Yearly Layoffs Trends by Country (Top 5)')
country_yearly_layoffs = data_filtered.groupby(['Year', 'Country'])['Laid_Off_Count'].sum().unstack(fill_value=0)
top_countries = country_yearly_layoffs.sum(axis=0).nlargest(5).index
top_country_data = country_yearly_layoffs[top_countries]
fig = px.line(top_country_data, labels={'value': 'Number of Layoffs', 'variable': 'Country'})
st.plotly_chart(fig)



st.subheader('Annual Layoffs by Industry (Top 5)')
industry_yearly_layoffs = data_filtered.groupby(['Year', 'Industry'])['Laid_Off_Count'].sum().unstack(fill_value=0)
top_industries = industry_yearly_layoffs.sum(axis=0).nlargest(5).index
top_industry_data = industry_yearly_layoffs[top_industries]
fig = px.area(top_industry_data, labels={'value': 'Number of Layoffs', 'variable': 'Industry'})
st.plotly_chart(fig)



st.subheader('Timeline of Layoff Events (Size Indicates Number of Layoffs)')
event_dates = data_filtered['Date']
layoff_counts = data_filtered['Laid_Off_Count'].fillna(0)
sizes = (layoff_counts / layoff_counts.max()) * 1000  # Normalize sizes for visibility
fig = px.scatter(x=event_dates, y=[1]*len(event_dates), size=sizes, labels={'x': 'Date'}, title='Timeline of Layoff Events', size_max=40)
fig.update_layout(yaxis={'visible': False, 'showticklabels': False})
st.plotly_chart(fig)


st.subheader('Correlation Between Funds Raised and Percentage of Workforce Laid Off')
clean_data = data_filtered.dropna(subset=['Funds_Raised', 'Percentage'])
fig = px.scatter(clean_data, x='Funds_Raised', y='Percentage', color='Industry',
                 labels={'Funds_Raised': 'Funds Raised (in millions)', 'Percentage': 'Percentage of Workforce Laid Off'},
                 title='Funds Raised vs. Percentage of Workforce Laid Off', trendline='ols')
fig.update_layout(xaxis_type='log', yaxis_type='log')
st.plotly_chart(fig)



st.subheader('Layoffs Distribution by Country and Industry')
country_industry_layoffs = data_filtered.groupby(['Country', 'Industry'])['Laid_Off_Count'].sum().reset_index()
fig = px.scatter(country_industry_layoffs, x='Country', y='Industry', size='Laid_Off_Count', color='Industry',
                 hover_name='Industry', size_max=60, title='Bubble Chart of Layoff Distribution by Country and Industry')
st.plotly_chart(fig)




st.subheader('Global Layoffs Visualization')
country_coords = {
    'United States': {'lat': 37.0902, 'lon': -95.7129},
    'Singapore': {'lat': 1.3521, 'lon': 103.8198},
    'India': {'lat': 20.5937, 'lon': 78.9629},
    'Australia': {'lat': -25.2744, 'lon': 133.7751},
    'Poland': {'lat': 51.9194, 'lon': 19.1451},
    'United Kingdom': {'lat': 55.3781, 'lon': -3.4360},
    'Sweden': {'lat': 60.1282, 'lon': 18.6435},
    'Israel': {'lat': 31.0461, 'lon': 34.8516},
    'Germany': {'lat': 51.1657, 'lon': 10.4515},
    'Norway': {'lat': 60.4720, 'lon': 8.4689},
    'Finland': {'lat': 61.9241, 'lon': 25.7482},
    'Canada': {'lat': 56.1304, 'lon': -106.3468},
    'Cayman Islands': {'lat': 19.3133, 'lon': -81.2546},
    'Czech Republic': {'lat': 49.8175, 'lon': 15.4730},
    'Lithuania': {'lat': 55.1694, 'lon': 23.8813},
    'Nigeria': {'lat': 9.0820, 'lon': 8.6753},
    'Japan': {'lat': 36.2048, 'lon': 138.2529},
    'Estonia': {'lat': 58.5953, 'lon': 25.0136},
    'Pakistan': {'lat': 30.3753, 'lon': 69.3451},
    'Austria': {'lat': 47.5162, 'lon': 14.5501},
    'Indonesia': {'lat': -0.7893, 'lon': 113.9213},
    'China': {'lat': 35.8617, 'lon': 104.1954},
    'France': {'lat': 46.2276, 'lon': 2.2137},
    'Netherlands': {'lat': 52.1326, 'lon': 5.2913},
    'Spain': {'lat': 40.4637, 'lon': -3.7492},
    'Brazil': {'lat': -14.2350, 'lon': -51.9253},
    'Switzerland': {'lat': 46.8182, 'lon': 8.2275},
    'New Zealand': {'lat': -40.9006, 'lon': 174.8860},
    'Hong Kong': {'lat': 22.3193, 'lon': 114.1694},
    'Kenya': {'lat': -0.0236, 'lon': 37.9062},
    'Luxembourg': {'lat': 49.8153, 'lon': 6.1296},
    'Saudi Arabia': {'lat': 23.8859, 'lon': 45.0792},
    'United Arab Emirates': {'lat': 23.4241, 'lon': 53.8478},
    'Philippines': {'lat': 12.8797, 'lon': 121.7740},
    'Mexico': {'lat': 23.6345, 'lon': -102.5528},
    'South Korea': {'lat': 35.9078, 'lon': 127.7669},
    'Vietnam': {'lat': 14.0583, 'lon': 108.2772},
    'Myanmar': {'lat': 21.9162, 'lon': 95.9560},
    'Egypt': {'lat': 26.8206, 'lon': 30.8025},
    'Colombia': {'lat': 4.5709, 'lon': -74.2973},
    'Chile': {'lat': -35.6751, 'lon': -71.5430},
    'Argentina': {'lat': -38.4161, 'lon': -63.6167},
    'Peru': {'lat': -9.1900, 'lon': -75.0152},
    'Turkey': {'lat': 38.9637, 'lon': 35.2433},
    'Russia': {'lat': 61.5240, 'lon': 105.3188},
    'South Africa': {'lat': -30.5595, 'lon': 22.9375},
    'Ghana': {'lat': 7.9465, 'lon': -1.0232},
    'Senegal': {'lat': 14.4974, 'lon': -14.4524}
}

data_filtered['lat'] = data_filtered['Country'].apply(lambda x: country_coords[x]['lat'] if x in country_coords else None)
data_filtered['lon'] = data_filtered['Country'].apply(lambda x: country_coords[x]['lon'] if x in country_coords else None)
country_layoffs = data_filtered.groupby('Country')['Laid_Off_Count'].sum().reset_index()
country_layoffs['lat'] = country_layoffs['Country'].apply(lambda x: country_coords[x]['lat'] if x in country_coords else None)
country_layoffs['lon'] = country_layoffs['Country'].apply(lambda x: country_coords[x]['lon'] if x in country_coords else None)
country_layoffs.dropna(subset=['lat', 'lon'], inplace=True)
fig = px.scatter_geo(country_layoffs, lat='lat', lon='lon', color='Laid_Off_Count', size='Laid_Off_Count',
                     hover_name='Country', projection="natural earth",color_continuous_scale='Viridis' )
st.plotly_chart(fig)




st.subheader('Annual Layoff Trends Across Major Industries')
major_industries = ['Tech', 'Finance', 'Healthcare', 'Manufacturing', 'Retail']
filtered_data = data_filtered[data_filtered['Industry'].isin(major_industries)]
industry_trends = filtered_data.groupby(['Year', 'Industry'])['Laid_Off_Count'].sum().unstack()
fig = px.line(industry_trends, labels={'value': 'Number of Layoffs', 'variable': 'Industry'})
st.plotly_chart(fig)



st.subheader('Distribution of Layoffs by Industry')
filtered_data = data_filtered[data_filtered['Laid_Off_Count'] <= data_filtered['Laid_Off_Count'].quantile(0.95)]
fig = px.box(filtered_data, x='Industry', y='Laid_Off_Count', points="all")
st.plotly_chart(fig)



st.subheader('Cumulative Layoff Trends Over Years')
annual_layoffs = data_filtered.groupby('Year')['Laid_Off_Count'].sum().cumsum()
fig = px.line(x=annual_layoffs.index, y=annual_layoffs, labels={'x': 'Year', 'y': 'Cumulative Number of Layoffs'}, title='Cumulative Layoff Trends Over Years')
st.plotly_chart(fig)




# st.subheader('Annual Layoff Trends Across Major Industries')
# major_industries = ['Tech', 'Finance', 'Healthcare', 'Manufacturing', 'Retail']
# filtered_data = data[data['Industry'].isin(major_industries)]
# industry_trends = filtered_data.groupby(['Year', 'Industry'])['Laid_Off_Count'].sum().unstack()

# # Convert DataFrame to a format suitable for Plotly
# fig = px.line(industry_trends, labels={'value': 'Number of Layoffs', 'variable': 'Industry'})
# fig.update_layout(title='Annual Layoff Trends Across Major Industries', xaxis_title='Year', yaxis_title='Number of Layoffs')
# st.plotly_chart(fig)



# st.subheader('Distribution of Layoffs by Industry')
# filtered_data = data[data['Laid_Off_Count'] <= data['Laid_Off_Count'].quantile(0.95)]
# fig, ax = plt.subplots(figsize=(12, 8))
# sns.boxplot(x='Industry', y='Laid_Off_Count', data=filtered_data)
# plt.title('Distribution of Layoffs by Industry')
# plt.xlabel('Industry')
# plt.ylabel('Number of Layoffs')
# plt.xticks(rotation=45)  # Rotate industry names for better readability
# sns.set_style("whitegrid")
# st.pyplot(fig)



# st.subheader('Cumulative Layoff Trends Over Years')
# annual_layoffs = data.groupby('Year')['Laid_Off_Count'].sum().cumsum()
# fig = px.line(x=annual_layoffs.index, y=annual_layoffs, labels={'x': 'Year', 'y': 'Cumulative Number of Layoffs'}, title='Cumulative Layoff Trends Over Years')
# st.plotly_chart(fig)



st.subheader('Layoff Trends by Industry Over Time')
industry_yearly = data.groupby(['Year', 'Industry'])['Laid_Off_Count'].sum().unstack(fill_value=0)
fig = px.area(industry_yearly, labels={'value': 'Number of Layoffs', 'variable': 'Industry'})
fig.update_layout(title='Layoff Trends by Industry Over Time', xaxis_title='Year', yaxis_title='Number of Layoffs', legend_title='Industry')
st.plotly_chart(fig)


st.subheader('Global Layoff Intensity by Country')
country_layoffs = data.groupby('Country')['Laid_Off_Count'].sum().reset_index()

fig = px.choropleth(country_layoffs,
                    locations="Country",
                    locationmode='country names',
                    color="Laid_Off_Count",
                    color_continuous_scale=px.colors.sequential.Plasma,
                    labels={'Laid_Off_Count': 'Number of Layoffs'},
                    title="Global Layoff Intensity by Country")
fig.update_layout(autosize=True, margin={"r":0, "t":30, "l":0, "b":0}, coloraxis_colorbar=dict(title="Layoffs"))
st.plotly_chart(fig)



st.subheader('Comparative Analysis of Layoffs and Funds Raised by Country')
country_aggregates = data.groupby('Country').agg({
    'Laid_Off_Count': 'sum',
    'Funds_Raised': 'sum'
}).reset_index()
country_aggregates = country_aggregates.sort_values(by='Laid_Off_Count', ascending=False)

# Plotting with dual axes using Plotly
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Bar(x=country_aggregates['Country'], y=country_aggregates['Laid_Off_Count'], name='Layoffs', marker_color='red'),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=country_aggregates['Country'], y=country_aggregates['Funds_Raised'], name='Funds Raised (millions)', marker_color='blue'),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="Comparative Analysis of Layoffs and Funds Raised by Country"
)

# Set x-axis title
fig.update_xaxes(title_text="Country")

# Set y-axes titles
fig.update_yaxes(title_text="<b>Layoffs</b>", secondary_y=False)
fig.update_yaxes(title_text="<b>Funds Raised (millions)</b>", secondary_y=True)

st.plotly_chart(fig)




import streamlit as st
import pandas as pd
import plotly.express as px

# Load and process data
@st.cache_data
def load_data():
    data = pd.read_csv('layoffs_data(4).csv')
    data['Date'] = pd.to_datetime(data['Date'])
    return data

data = load_data()
data['Month'] = data['Date'].dt.to_period('M').dt.to_timestamp()  # Convert to Timestamp immediately

# Time Series Plot of Layoffs
st.subheader("Time Series Plot of Layoffs")
monthly_layoffs = data.groupby('Month')['Laid_Off_Count'].sum().reset_index(name='Layoffs')
fig1 = px.line(monthly_layoffs, x='Month', y='Layoffs', title='Monthly Layoffs Over Time')
st.plotly_chart(fig1)

# Bar Chart of Industries Affected
st.subheader("Bar Chart of Industries Affected")
# Sum layoffs by industry
industry_layoffs = data.groupby('Industry')['Laid_Off_Count'].sum().sort_values(ascending=False)
fig2 = px.bar(industry_layoffs, x=industry_layoffs.index, y=industry_layoffs.values, title='Layoffs by Industry')
st.plotly_chart(fig2)

# # (wrong code) Bar Chart of Industries Affected
# st.subheader("Bar Chart of Industries Affected")
# industry_layoffs = data['Industry'].value_counts()
# fig2 = px.bar(industry_layoffs, x=industry_layoffs.index, y=industry_layoffs.values, title='Layoffs by Industry')
# st.plotly_chart(fig2)

# Funds Raised vs. Layoffs Scatter Plot
st.subheader("Funds Raised vs. Layoffs Scatter Plot")
fig3 = px.scatter(data, x='Funds_Raised', y='Percentage', color='Industry', title='Funds Raised vs. Layoff Rate',
                  hover_data=['Company'])
fig3.update_traces(marker=dict(size=10, opacity=0.5))
st.plotly_chart(fig3)

# Pie Chart of Company Stages
st.subheader("Pie Chart of Company Stages")
stage_counts = data['Stage'].value_counts()
fig4 = px.pie(values=stage_counts.values, names=stage_counts.index, title='Company Stages')
st.plotly_chart(fig4)

# Stacked Bar Chart of Layoffs Over Time by Industry
st.subheader("Stacked Bar Chart of Layoffs Over Time by Industry")
layoffs_by_industry_time = pd.crosstab(data['Month'], data['Industry'])
fig5 = px.bar(layoffs_by_industry_time, barmode='stack', title='Layoffs Over Time by Industry')
st.plotly_chart(fig5)

# st.write("Layoffs Dashboard Updated Successfully")
