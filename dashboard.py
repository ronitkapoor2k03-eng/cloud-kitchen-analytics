import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Cloud Kitchen Analytics", layout="wide")

st.title("Cloud Kitchen Customer Analytics Dashboard")
st.markdown("Data-Driven Insights for Business Growth")
st.markdown("---")

@st.cache_data
def load_data():
    data = {
        'Customer ID': [f'C{i:03d}' for i in range(1, 101)],
        'Age': [25,32,28,40,35,29,45,31,27,38,24,36,41,30,34,26,39,33,28,37,29,31,42,35,27,38,26,34,43,36,28,30,41,37,25,33,44,31,29,38,27,35,42,36,28,39,26,34,41,37,29,31,43,35,27,38,25,33,44,31,28,30,42,36,26,34,41,37,27,35,43,36,29,39,26,34,42,37,28,30,41,36,27,35,44,31,29,38,26,34,41,37,28,30,42,36,27,35,44,31],
        'Area': ['Juhu','Bandra','Andheri','Dadar','Vile Parle','Khar','Churchgate','Matunga','Mahim','Wadala','Bandra','Andheri','Juhu','Dadar','Vile Parle','Khar','Churchgate','Matunga','Mahim','Wadala','Juhu','Bandra','Andheri','Dadar','Vile Parle','Khar','Churchgate','Matunga','Mahim','Wadala','Juhu','Bandra','Andheri','Dadar','Vile Parle','Khar','Churchgate','Matunga','Mahim','Wadala','Juhu','Bandra','Andheri','Dadar','Vile Parle','Khar','Churchgate','Matunga','Mahim','Wadala','Juhu','Bandra','Andheri','Dadar','Vile Parle','Khar','Churchgate','Matunga','Mahim','Wadala','Juhu','Bandra','Andheri','Dadar','Vile Parle','Khar','Churchgate','Matunga','Mahim','Wadala','Juhu','Bandra','Andheri','Dadar','Vile Parle','Khar','Churchgate','Matunga','Mahim','Wadala','Juhu','Bandra','Andheri','Dadar','Vile Parle','Khar','Churchgate','Matunga','Mahim','Wadala','Juhu','Bandra','Andheri','Dadar','Vile Parle','Khar','Churchgate','Matunga','Mahim','Wadala'],
        'Income': [45000,60000,52000,30000,75000,48000,28000,67000,54000,80000,42000,62000,35000,70000,58000,46000,31000,72000,55000,78000,50000,61000,32000,76000,53000,82000,47000,59000,33000,77000,51000,62000,34000,79000,44000,60000,29000,68000,56000,81000,50000,63000,31000,78000,54000,82000,46000,58000,33000,80000,51000,62000,32000,76000,53000,83000,45000,60000,30000,69000,52000,61000,34000,77000,47000,59000,31000,79000,49000,63000,33000,78000,56000,82000,46000,58000,32000,80000,52000,61000,34000,77000,48000,63000,30000,69000,56000,82000,47000,58000,33000,80000,52000,61000,34000,77000,48000,63000,30000,69000],
        'Visits': [12,18,10,8,20,14,6,22,15,25,11,19,9,21,16,13,7,23,14,24,12,17,8,22,13,25,11,18,9,23,13,17,8,24,10,19,7,22,15,25,12,18,8,23,14,25,11,17,9,24,13,18,8,22,14,25,11,19,7,22,13,17,8,23,12,18,9,24,11,18,8,23,15,25,12,17,9,24,13,17,8,23,12,18,7,22,15,25,11,17,9,24,13,17,8,23,12,18,7,22],
        'Clicks': [30,45,20,15,60,35,10,55,40,70,25,50,18,65,42,33,12,68,38,72,28,48,14,60,34,75,29,46,16,69,31,44,17,71,27,52,11,63,39,74,30,47,13,67,36,76,28,45,15,73,32,49,14,64,35,77,29,51,12,66,30,43,16,70,31,46,14,72,28,48,13,70,38,75,30,44,16,73,31,45,14,70,29,48,11,66,39,76,28,45,15,73,30,43,16,70,29,48,11,66],
        'Orders': [3,5,2,1,6,3,1,7,4,8,2,5,2,6,4,3,1,7,3,8,3,5,1,6,3,9,2,5,1,7,3,4,2,8,2,5,1,6,4,9,3,5,1,7,3,9,2,4,1,8,3,5,1,6,3,9,2,5,1,6,3,4,2,7,3,5,1,8,2,5,1,7,4,9,3,4,1,8,3,4,1,7,3,5,1,6,4,9,2,4,1,8,3,4,1,7,2,5,1,6],
        'Average Order Value': [1500,2000,1800,1200,2500,1700,1000,2300,1900,2600,1400,2100,1300,2400,2000,1600,1100,2500,1800,2700,1500,2100,1200,2400,1700,2800,1500,2000,1300,2600,1600,2100,1200,2700,1400,2200,1000,2300,1900,2800,1600,2100,1200,2500,1800,2900,1500,2000,1300,2700,1600,2100,1200,2400,1700,2900,1500,2200,1000,2300,1600,2000,1300,2600,1500,2000,1000,2700,1400,2100,1300,2500,1900,2800,1500,2000,1000,2700,1600,2100,1300,2600,1500,2100,1000,2300,1900,2900,1500,2000,1300,2700,1600,2100,1300,2600,1500,2100,1000,2300],
        'Total Spend': [4500,10000,3600,1200,15000,5100,1000,16100,7600,20800,2800,10500,2600,14400,8000,4800,1100,17500,5400,21600,4500,10500,1200,14400,5100,25200,3000,10000,1300,18200,4800,8400,2400,21600,2800,11000,1000,13800,7600,25200,4800,10500,1200,17500,5400,26100,3000,8000,1300,21600,4800,10500,1200,14400,5100,26100,3000,11000,1000,13800,4800,8000,2600,18200,4500,10000,1200,21600,2800,10500,2600,17500,7600,25200,4800,8000,1200,21600,4800,8400,2600,18200,3000,10500,1000,13800,7600,26100,3000,8000,1300,21600,4800,8400,2600,18200,3000,10500,1000,13800],
        'Customer Segment': ['Low Income','Medium Income','Medium Income','Low Income','High Income','Low Income','Low Income','Medium Income','Medium Income','High Income','Low Income','Medium Income','Low Income','Medium Income','Medium Income','Low Income','Low Income','High Income','Medium Income','High Income','Low Income','Medium Income','Low Income','High Income','Medium Income','High Income','Low Income','Medium Income','Low Income','High Income','Medium Income','Medium Income','Low Income','High Income','Low Income','Medium Income','Low Income','Medium Income','Medium Income','High Income','Low Income','Medium Income','Low Income','High Income','Medium Income','High Income','Low Income','Medium Income','Low Income','High Income','Medium Income','Medium Income','Low Income','High Income','Medium Income','High Income','Low Income','Medium Income','Low Income','Medium Income','Medium Income','Medium Income','Low Income','High Income','Low Income','Medium Income','Low Income','High Income','Low Income','Medium Income','Low Income','High Income','Medium Income','High Income','Low Income','Medium Income','Low Income','High Income','Medium Income','Medium Income','Low Income','High Income','Low Income','Medium Income','Low Income','Medium Income','Medium Income','High Income','Low Income','Medium Income','Low Income','High Income','Medium Income','Medium Income','Low Income','High Income','Low Income','Medium Income','Low Income','Medium Income'],
        'Engagement Rate': [2.50,2.50,2.00,1.88,3.00,2.50,1.67,2.50,2.67,2.80,2.27,2.63,2.00,3.10,2.63,2.54,1.71,2.96,2.71,3.00,2.33,2.82,1.75,2.73,2.62,3.00,2.64,2.56,1.78,3.00,2.38,2.59,2.13,2.96,2.70,2.74,1.57,2.86,2.60,2.96,2.50,2.61,1.63,2.91,2.57,3.04,2.55,2.65,1.67,3.04,2.46,2.72,1.75,2.91,2.50,3.08,2.64,2.68,1.71,3.00,2.31,2.53,2.00,3.04,2.58,2.56,1.56,3.00,2.55,2.67,1.88,3.00,2.53,3.00,2.50,2.59,1.56,3.04,2.38,2.65,2.00,3.04,2.42,2.67,1.57,3.00,2.60,3.04,2.55,2.65,1.67,3.04,2.31,2.53,2.00,3.04,2.42,2.67,1.57,3.00],
        'Conversion Rate': [25,28,20,13,30,21,17,32,27,32,18,26,22,29,25,23,14,30,21,33,25,29,13,27,23,36,18,28,11,30,23,24,25,33,20,26,14,27,27,36,25,28,13,30,21,36,18,24,11,33,23,28,13,27,21,36,18,26,14,27,23,24,25,30,25,28,11,33,18,28,25,30,27,36,25,24,11,33,23,24,25,30,18,28,14,27,27,36,18,24,11,33,23,24,25,30,18,28,14,27],
        'Customer Value': [375.00,555.56,360.00,150.00,750.00,364.29,166.67,731.82,506.67,832.00,254.55,552.63,288.89,685.71,500.00,369.23,157.14,760.87,385.71,900.00,375.00,617.65,150.00,654.55,392.31,1008.00,272.73,555.56,144.44,791.30,369.23,494.12,300.00,900.00,280.00,578.95,142.86,627.27,506.67,1008.00,400.00,583.33,150.00,760.87,385.71,1044.00,272.73,470.59,144.44,900.00,369.23,583.33,150.00,654.55,364.29,1044.00,272.73,578.95,142.86,627.27,369.23,470.59,325.00,791.30,375.00,555.56,133.33,900.00,254.55,583.33,162.50,760.87,506.67,1008.00,375.00,470.59,133.33,900.00,369.23,494.12,325.00,791.30,250.00,583.33,142.86,627.27,506.67,1044.00,272.73,470.59,144.44,900.00,369.23,470.59,325.00,791.30,250.00,583.33,142.86,627.27]
    }
    return pd.DataFrame(data)

df = load_data()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Section", 
                        ["Overview", 
                         "Descriptive Analytics", 
                         "Diagnostic Analytics", 
                         "Predictive Analytics", 
                         "Prescriptive Analytics"])

if page == "Overview":
    st.header("Dashboard Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", len(df))
    with col2:
        st.metric("Average Customer Value", f"Rs {df['Customer Value'].mean():.2f}")
    with col3:
        st.metric("Average Conversion Rate", f"{df['Conversion Rate'].mean():.1f}%")
    with col4:
        st.metric("Total Revenue", f"Rs {df['Total Spend'].sum():,.0f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Customer Segment Distribution")
        fig = px.pie(df, names='Customer Segment', title='Customer Segments', hole=0.3)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top Areas by Customer Value")
        area_value = df.groupby('Area')['Customer Value'].mean().sort_values(ascending=False).head(10)
        fig = px.bar(x=area_value.values, y=area_value.index, orientation='h', title='Average Customer Value by Area')
        st.plotly_chart(fig, use_container_width=True)

elif page == "Descriptive Analytics":
    st.header("Descriptive Analytics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Customer Value Distribution")
        fig = px.histogram(df, x='Customer Value', nbins=25, title='Distribution of Customer Value')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Engagement Rate Distribution")
        fig = px.histogram(df, x='Engagement Rate', nbins=20, title='Distribution of Engagement Rate')
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Conversion Rate Distribution")
        fig = px.histogram(df, x='Conversion Rate', nbins=20, title='Distribution of Conversion Rate (%)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Income by Customer Segment")
        fig = px.box(df, x='Customer Segment', y='Income', title='Income Distribution by Segment')
        st.plotly_chart(fig, use_container_width=True)

elif page == "Diagnostic Analytics":
    st.header("Diagnostic Analytics")
    
    st.subheader("Correlation Analysis")
    numeric_cols = ['Age', 'Income', 'Visits', 'Clicks', 'Orders', 'Average Order Value', 'Total Spend', 'Engagement Rate', 'Conversion Rate', 'Customer Value']
    corr_matrix = df[numeric_cols].corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Heatmap", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Clicks vs Customer Value")
        fig = px.scatter(df, x='Clicks', y='Customer Value', color='Customer Segment', title='Clicks vs Customer Value')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Income vs Customer Value")
        fig = px.scatter(df, x='Income', y='Customer Value', color='Customer Segment', title='Income vs Customer Value')
        st.plotly_chart(fig, use_container_width=True)

elif page == "Predictive Analytics":
    st.header("Predictive Analytics")
    
    feature_cols = ['Age', 'Income', 'Visits', 'Clicks', 'Orders', 'Average Order Value', 'Engagement Rate']
    X = df[feature_cols]
    y_value = df['Customer Value']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_value, test_size=0.2, random_state=42)
    
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_train, y_train)
    y_pred = rf_reg.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    st.metric("Model R2 Score", f"{r2:.3f}")
    
    fig = px.scatter(x=y_test, y=y_pred, title='Actual vs Predicted Customer Value')
    fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', name='Perfect Prediction', line=dict(color='red', dash='dash')))
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Feature Importance")
    importance_value = pd.DataFrame({'Feature': feature_cols, 'Importance': rf_reg.feature_importances_}).sort_values('Importance', ascending=True)
    fig = px.bar(importance_value, x='Importance', y='Feature', orientation='h', title='What drives Customer Value?')
    st.plotly_chart(fig, use_container_width=True)

else:
    st.header("Prescriptive Analytics")
    
    feature_cols = ['Age', 'Income', 'Visits', 'Clicks', 'Orders', 'Average Order Value', 'Engagement Rate']
    X = df[feature_cols]
    y_value = df['Customer Value']
    X_train, X_test, y_train, y_test = train_test_split(X, y_value, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    baseline = df[feature_cols].mean().values
    baseline_value = model.predict([baseline])[0]
    
    st.subheader("What-If Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        clicks_change = st.slider("Clicks Change (%)", -50, 100, 0, 5)
        orders_change = st.slider("Orders Change (%)", -50, 100, 0, 5)
    with col2:
        engagement_change = st.slider("Engagement Rate Change (%)", -50, 100, 0, 5)
        visits_change = st.slider("Visits Change (%)", -50, 100, 0, 5)
    
    scenario = baseline.copy()
    scenario[feature_cols.index('Clicks')] *= (1 + clicks_change/100)
    scenario[feature_cols.index('Orders')] *= (1 + orders_change/100)
    scenario[feature_cols.index('Engagement Rate')] *= (1 + engagement_change/100)
    scenario[feature_cols.index('Visits')] *= (1 + visits_change/100)
    scenario_value = model.predict([scenario])[0]
    improvement = ((scenario_value - baseline_value) / baseline_value) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Baseline Value", f"Rs {baseline_value:.2f}")
    with col2:
        st.metric("New Value", f"Rs {scenario_value:.2f}")
    with col3:
        st.metric("Improvement", f"{improvement:.1f}%")
    
    st.subheader("Recommendations by Segment")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Low Value Customers (Below Rs 300)**")
        st.markdown("- Send personalized discount offers")
        st.markdown("- Implement loyalty program")
        st.markdown("- Email re-engagement campaigns")
        st.markdown("")
        st.markdown("**Medium Value Customers (Rs 300 to 700)**")
        st.markdown("- Upsell premium combos")
        st.markdown("- Introduce subscription model")
        st.markdown("- Cross-sell complementary items")
    with col2:
        st.markdown("**High Value Customers (Above Rs 700)**")
        st.markdown("- Exclusive loyalty rewards")
        st.markdown("- VIP early access")
        st.markdown("- Personalized chef recommendations")
