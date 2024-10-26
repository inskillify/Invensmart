import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
from streamlit_folium import folium_static
import folium
import random

# Set page layout
st.set_page_config(page_title="InvenSmart India Dashboard", layout="wide")

# Generate dynamic sample data
def generate_sample_data(n_records=100, start_date=None):
    if start_date is None:
        start_date = datetime.now() - timedelta(days=60)
    
    categories = ['Electronics', 'Clothing', 'Food', 'Books', 'Home Goods']
    products = [f'Product_{i}' for i in range(1, n_records + 1)]
    
    # Indian states with major cities for map locations
    store_bases = {
        'Delhi': (28.7041, 77.1025),
        'Mumbai': (19.0760, 72.8777),
        'Bengaluru': (12.9716, 77.5946),
        'Hyderabad': (17.3850, 78.4867),
        'Chennai': (13.0827, 80.2707),
        'Kolkata': (22.5726, 88.3639),
        'Ahmedabad': (23.0225, 72.5714),
        'Pune': (18.5204, 73.8567),
        'Jaipur': (26.9124, 75.7873),
        'Lucknow': (26.8467, 80.9462)
    }
    
    # Generate daily data for the past 60 days
    all_data = []
    current_date = start_date
    
    while current_date <= datetime.now():
        for _ in range(n_records // 60):  # Distribute records across days
            # Add small random variation to store locations
            store_locations = {
                store: (
                    base[0] + random.uniform(-0.002, 0.002),
                    base[1] + random.uniform(-0.002, 0.002)
                )
                for store, base in store_bases.items()
            }
            
            location = random.choice(list(store_locations.keys()))
            lat, lon = store_locations[location]
            
            # Generate dynamic sales and stock data
            base_sales = random.uniform(100, 10000)
            # Add seasonal and weekly patterns
            day_factor = 1 + 0.2 * np.sin(current_date.weekday() * np.pi / 7)  # Weekly pattern
            month_factor = 1 + 0.3 * np.sin(current_date.month * np.pi / 6)    # Seasonal pattern
            
            sales = base_sales * day_factor * month_factor
            
            record = {
                'Product_ID': random.choice(products),
                'Category': random.choice(categories),
                'Location_Name': location,
                'Stock_Level': random.randint(10, 1000),
                'Sales_Volume': sales,
                'Last_Restock_Date': current_date.strftime('%d-%m-%Y'),
                'Latitude': lat,
                'Longitude': lon
            }
            all_data.append(record)
        
        current_date += timedelta(days=1)
    
    return pd.DataFrame(all_data)

# Utility functions
def generate_sales_insights(df):
    insights = []
    
    # Sales trend analysis with time-based components
    recent_sales = df.sort_values('Last_Restock_Date')
    sales_trend = recent_sales.groupby('Last_Restock_Date')['Sales_Volume'].sum().pct_change().mean()
    
    if sales_trend > 0:
        insights.append(f"📈 Sales are trending upward with {sales_trend:.1%} average daily growth")
    else:
        insights.append(f"📉 Sales are trending downward with {abs(sales_trend):.1%} average daily decline")
    
    # Dynamic category performance
    category_performance = df.groupby('Category')['Sales_Volume'].agg(['sum', 'mean'])
    top_category = category_performance['sum'].idxmax()
    category_sales = category_performance.loc[top_category, 'sum']
    insights.append(f"🏆 Best performing category: {top_category} with ₹{category_sales:,.2f} in sales")
    
    # Location-based insights
    location_performance = df.groupby('Location_Name')['Sales_Volume'].sum()
    best_location = location_performance.idxmax()
    insights.append(f"📍 Top performing location: {best_location} with ₹{location_performance[best_location]:,.2f} in sales")
    
    return insights

def generate_recommendations(df):
    recommendations = []
    
    # Dynamic inventory optimization
    stock_sales_ratio = df.groupby('Product_ID').agg({
        'Stock_Level': 'mean',
        'Sales_Volume': 'mean'
    })
    stock_sales_ratio['ratio'] = stock_sales_ratio['Stock_Level'] / stock_sales_ratio['Sales_Volume']
    
    high_stock_items = stock_sales_ratio[stock_sales_ratio['ratio'] > stock_sales_ratio['ratio'].quantile(0.75)]
    if not high_stock_items.empty:
        recommendations.append(f"🔄 Consider reducing stock for {len(high_stock_items)} items with high stock-to-sales ratio")
    
    # Location-based recommendations
    location_metrics = df.groupby('Location_Name').agg({
        'Sales_Volume': 'mean',
        'Stock_Level': 'mean'
    })
    
    low_stock_locations = location_metrics[location_metrics['Stock_Level'] < location_metrics['Stock_Level'].mean() * 0.8]
    if not low_stock_locations.empty:
        recommendations.append(f"📦 Restock needed at {len(low_stock_locations)} locations with low inventory")
    
    return recommendations

def calculate_metrics(df):
    # Enhanced metrics with time-based components
    metrics = {
        'total_sales': df['Sales_Volume'].sum(),
        'avg_daily_sales': df.groupby('Last_Restock_Date')['Sales_Volume'].sum().mean(),
        'stock_turnover': df['Sales_Volume'].sum() / df['Stock_Level'].sum(),
        'low_stock_items': len(df[df['Stock_Level'] < df['Stock_Level'].mean() * 0.2])
    }
    
    # Add trend calculations
    daily_sales = df.groupby('Last_Restock_Date')['Sales_Volume'].sum()
    metrics['sales_trend'] = daily_sales.pct_change().mean()
    
    return metrics

def create_store_map(df):
    store_metrics = df.groupby(['Location_Name']).agg({
        'Latitude': 'first',
        'Longitude': 'first',
        'Sales_Volume': 'sum',
        'Stock_Level': 'mean'
    }).reset_index()
    
    # Create map centered on India
    m = folium.Map(
        location=[20.5937, 78.9629],  # Centered on India
        zoom_start=5
    )
    
    for idx, row in store_metrics.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"""
                <b>{row['Location_Name']}</b><br>
                Sales: ₹{row['Sales_Volume']:,.2f}<br>
                Avg Stock: {row['Stock_Level']:,.0f}
            """,
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m)
    
    return m

def get_top_selling_products(df, n=5):
    top_products = df.groupby('Product_ID')['Sales_Volume'].sum().nlargest(n)
    return top_products

# Load or generate data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/hyperlocal_inventory_data_updated.csv')
    except FileNotFoundError:
        st.info("InvenSmart")
        df = generate_sample_data()
    
    # Convert 'Last_Restock_Date' to datetime format with day-first format
    df['Last_Restock_Date'] = pd.to_datetime(df['Last_Restock_Date'], format='%d-%m-%Y', dayfirst=True, errors='coerce')
    return df

# Load data
df = load_data()

# Sidebar
with st.sidebar:
    st.title("InvenSmart India Dashboard")
    st.markdown("## Navigation")
    page = st.radio("Go to", ["Dashboard", "Analytics", "Store Map", "AI Insights", "Recommendations"])
    
    st.markdown("### Filters")
    date_range = st.selectbox("Select Date Range", ["Last 7 Days", "Last 30 Days", "Last 60 Days"])
    selected_category = st.selectbox("Category Filter", ["All"] + list(df['Category'].unique()))

# Apply filters
today = pd.Timestamp.now()
if date_range == "Last 7 Days":
    df_filtered = df[df['Last_Restock_Date'] >= today - pd.DateOffset(days=7)]
elif date_range == "Last 30 Days":
    df_filtered = df[df['Last_Restock_Date'] >= today - pd.DateOffset(days=30)]
else:
    df_filtered = df[df['Last_Restock_Date'] >= today - pd.DateOffset(days=60)]

if selected_category != "All":
    df_filtered = df_filtered[df_filtered['Category'] == selected_category]

# Dashboard Page
if page == "Dashboard":
    st.title("Dashboard")
    
    # Calculate metrics
    metrics = calculate_metrics(df_filtered)
    
    # Display key metrics
    st.metric("Total Sales", f"₹{metrics['total_sales']:,.2f}")
    st.metric("Average Daily Sales", f"₹{metrics['avg_daily_sales']:.2f}")
    st.metric("Stock Turnover Ratio", f"{metrics['stock_turnover']:.2f}")
    st.metric("Low Stock Items", metrics['low_stock_items'])
    
    # Display pie chart for category distribution
    category_pie_chart = px.pie(df_filtered, names="Category", values="Sales_Volume", title="Sales Volume by Category")
    st.plotly_chart(category_pie_chart, use_container_width=True)

    # Display bar chart for stock levels by location
    stock_bar_chart = px.bar(df_filtered, x="Location_Name", y="Stock_Level", color="Category", title="Stock Levels by Location")
    st.plotly_chart(stock_bar_chart, use_container_width=True)
    
    # Display top-selling products
    st.subheader("Top Selling Products")
    top_products = get_top_selling_products(df_filtered, n=5)
    top_products_df = top_products.reset_index()
    
    # Calculate the percentage of total sales
    total_sales_filtered = metrics['total_sales']
    top_products_df['Percentage of Demand'] = (top_products_df['Sales_Volume'] / total_sales_filtered * 100).round(2)
    
    st.write(top_products_df)

# Analytics Page
elif page == "Analytics":
    st.title("Analytics")
    fig = px.histogram(df_filtered, x="Category", y="Sales_Volume", color="Location_Name", title="Sales by Category and Location")
    st.plotly_chart(fig)

# Store Map Page
elif page == "Store Map":
    st.title("Store Map")
    m = create_store_map(df_filtered)
    folium_static(m)

# AI Insights Page
elif page == "AI Insights":
    st.title("AI Sales Insights")
    insights = generate_sales_insights(df_filtered)
    
    # Map for top performing location insights
    st.subheader("Sales Performance Insights by Location")
    m = create_store_map(df_filtered)
    folium_static(m)
    
    for insight in insights:
        st.write(insight)

# Recommendations Page
elif page == "Recommendations":
    st.title("Inventory Recommendations")
    recommendations = generate_recommendations(df_filtered)
    
    # Display each recommendation in a styled box
    for recommendation in recommendations:
        st.markdown(f"""
        <div style="padding: 10px; border: 1px solid #4CAF50; border-radius: 5px; margin: 5px 0;">
            {recommendation}
        </div>
        """, unsafe_allow_html=True)
