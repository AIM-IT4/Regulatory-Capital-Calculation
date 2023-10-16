#!/usr/bin/env python
# coding: utf-8

# In[ ]:




# # Regulatory Capital Calculation Project
# ## Project Outline
# 1. **Data Collection and Preprocessing**
#     - Data Sources
#     - Data Cleaning
#     - Data Storage
# 2. **Risk Calculations**
#     - Market Risk
#     - Credit Risk
#     - Operational Risk
# 3. **Capital Calculation**
#     - Tier 1 and Tier 2 Capital
#     - Risk-Weighted Assets (RWA)
#     - Capital Ratios
# 4. **Reporting**
#     - Compliance Reports
#     - Dashboard
# 5. **Testing and Validation**
#     - Backtesting
#     - Sensitivity Analysis
# 6. **Documentation**
#     - User Guide
#     - Technical Documentation
# 
# ## Tools and Libraries
# - QuantLib for various financial calculations.
# - SQL/NoSQL Database for data storage.
# - Python/R for data manipulation and analysis.
# - Tableau/Power BI for dashboards and reporting.

# ## Focus: Credit Risk
# For this project, we will focus specifically on the Credit Risk component. The following are the key steps we will undertake:
# 1. **Generate Synthetic Data**: Create synthetic data for a portfolio of loans or bonds, including features like credit rating, maturity, coupon rate, etc.
# 2. **Credit Risk Models**: Implement credit risk models like CreditMetrics or KMV-Merton.
# 3. **Credit VaR Calculation**: Calculate Credit Value at Risk (VaR) based on the chosen model.
# 4. **Sensitivity Analysis**: Analyze how changes in credit ratings, interest rates, and other factors affect Credit VaR.
# 5. **Reporting**: Generate reports showing the Credit VaR and other key metrics.

# In[ ]:


import pandas as pd
import numpy as np
import random

# Generate synthetic data for a portfolio of loans or bonds
np.random.seed(42)
n = 100  # Number of loans or bonds in the portfolio

# Features of the portfolio
credit_ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C']
maturities = np.random.randint(1, 31, n)  # Maturity in years
coupon_rates = np.random.uniform(0.01, 0.2, n)  # Coupon rate in percentage
face_values = np.random.randint(1000, 10000, n)  # Face value of the bond or loan

# Generate the DataFrame
portfolio_df = pd.DataFrame({
    'Credit Rating': np.random.choice(credit_ratings, n),
    'Maturity (Years)': maturities,
    'Coupon Rate (%)': coupon_rates * 100,
    'Face Value ($)': face_values
})

portfolio_df.head()

# In[ ]:


!pip install -q QuantLib-Python

# ## Credit Risk Modeling using QuantLib
# We will use QuantLib to implement a credit risk model for our synthetic portfolio. Specifically, we'll use the KMV-Merton model to assess the credit risk.

# In[ ]:


import QuantLib as ql
from scipy.stats import norm
# Function to calculate the default probability using the KMV-Merton model
def calculate_default_probability(equity, equity_volatility, face_value, risk_free_rate, maturity):
    # Create the option object
    payoff = ql.PlainVanillaPayoff(ql.Option.Put, face_value)
    exercise = ql.EuropeanExercise(ql.NullCalendar().advance(ql.Date.todaysDate(), ql.Period(int(maturity), ql.Years)))
    option = ql.VanillaOption(payoff, exercise)

    # Create the Black-Scholes-Merton process
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(equity))
    rate_handle = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(risk_free_rate)), ql.Actual360()))
    vol_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(0, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(equity_volatility)), ql.Actual360()))
    bsm_process = ql.BlackScholesMertonProcess(spot_handle, ql.YieldTermStructureHandle(ql.FlatForward(0, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(0.0)), ql.Actual360())), rate_handle, vol_handle)

    # Calculate the option price using the BSM model
    engine = ql.AnalyticEuropeanEngine(bsm_process)
    option.setPricingEngine(engine)
    option_price = option.NPV()

    # Calculate the distance to default
    distance_to_default = (np.log(equity / face_value) + (risk_free_rate - 0.5 * equity_volatility ** 2) * maturity) / (equity_volatility * np.sqrt(maturity))

    # Calculate the default probability using scipy's norm function
    default_probability = norm.cdf(-distance_to_default)

    return default_probability

# Test the function
calculate_default_probability(5000, 0.2, 6000, 0.01, 5)

# In[ ]:


# Apply the KMV-Merton model to the synthetic portfolio
# For simplicity, let's assume some constant values for equity and equity_volatility for each loan/bond
equity = 5000  # Hypothetical equity value
equity_volatility = 0.2  # Hypothetical equity volatility
risk_free_rate = 0.01  # Hypothetical risk-free rate
# Calculate default probabilities for the portfolio
portfolio_df['Default Probability'] = portfolio_df.apply(lambda row: calculate_default_probability(equity, equity_volatility, row['Face Value ($)'], risk_free_rate, row['Maturity (Years)']), axis=1)
portfolio_df.head()

# ## Sensitivity Analysis
# In this section, we will perform a sensitivity analysis to understand how changes in various parameters like credit ratings, interest rates, and other factors affect the default probabilities.

# In[ ]:


import matplotlib.pyplot as plt
# Function to perform sensitivity analysis on default probability with respect to face value
def sensitivity_analysis_face_value(equity, equity_volatility, risk_free_rate, maturities):
    face_values = np.linspace(4000, 8000, 50)
    default_probabilities = [calculate_default_probability(equity, equity_volatility, fv, risk_free_rate, maturities) for fv in face_values]

    plt.figure(figsize=(10, 6))
    plt.plot(face_values, default_probabilities, label=f'Maturity: {maturities} years')
    plt.xlabel('Face Value ($)')
    plt.ylabel('Default Probability')
    plt.title('Sensitivity of Default Probability to Face Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# Perform sensitivity analysis for a 5-year and 10-year maturity
sensitivity_analysis_face_value(5000, 0.2, 0.01, 5)
sensitivity_analysis_face_value(5000, 0.2, 0.01, 10)

# ## Reporting for Compliance
# In this section, we will generate reports that can be used for regulatory compliance. These reports will include key metrics like Value-at-Risk (VaR), Expected Shortfall, and the calculated default probabilities.

# In[ ]:


# Generate a summary report for compliance
summary_report = portfolio_df.groupby('Credit Rating').agg({
    'Default Probability': ['mean', 'std'],
    'Face Value ($)': ['sum'],
    'Maturity (Years)': ['mean']
}).reset_index()
summary_report.columns = ['Credit Rating', 'Mean Default Probability', 'Std Dev Default Probability', 'Total Face Value ($)', 'Mean Maturity (Years)']
summary_report

# In[ ]:


# Save the synthetic portfolio data to a CSV file
portfolio_csv_file = 'synthetic_portfolio_data.csv'
portfolio_df.to_csv(portfolio_csv_file, index=False)
