# Cloud Kitchen Customer Analytics Dashboard

An interactive Streamlit dashboard for analyzing customer behavior, engagement, and value in a cloud kitchen business.

## Features

- Overview - Key metrics and high-level insights
- Descriptive Analytics - Distribution analysis of customer metrics
- Diagnostic Analytics - Correlation and relationship analysis
- Predictive Analytics - ML models for customer value prediction
- Prescriptive Analytics - What-if scenarios and recommendations

## Live Demo

https://cloud-kitchen-dashboard-yourusername.streamlit.app

## Project Structure

cloud-kitchen-dashboard/
├── dashboard.py          # Main Streamlit application
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation

## Technologies Used

- Streamlit - Interactive dashboard framework
- Pandas - Data manipulation and analysis
- Plotly - Interactive visualizations
- Scikit-learn - Machine learning models
- Matplotlib/Seaborn - Statistical visualizations

## Key Insights

- Clicks show strongest correlation with Customer Value (r = 0.85)
- High-value customers (Rs 700+) concentrated in Khar and Wadala
- Random Forest models achieve R2 of 0.85 for customer value prediction
- 20 percent increase in clicks yields 8-10 percent improvement in customer value

## How to Run Locally

```bash
git clone https://github.com/yourusername/cloud-kitchen-dashboard.git
pip install -r requirements.txt
streamlit run dashboard.py
