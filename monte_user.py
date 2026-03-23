import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t, norm, chi2

# Page Config
st.set_page_config(page_title="Monte Carlo Risk Engine", layout="wide")

st.title("Monte Carlo Risk Engine (Student's t)")

def run_backtest_ui(returns, confidence_level=0.95):
    st.subheader("Backtesting: Kupiec POF Test")
    
    z_score = norm.ppf(confidence_level)
    daily_vol = returns.std()
    var_cutoff_percent = daily_vol * z_score

    actual_failures = returns[returns < -var_cutoff_percent]
    num_failures = len(actual_failures)
    total_days = len(returns)
    num_success = total_days - num_failures

    expected_failure_rate = 1 - confidence_level
    expected_success_rate = confidence_level
    actual_failure_rate = num_failures / total_days
    actual_success_rate = 1 - actual_failure_rate
    expected_failures = total_days * expected_failure_rate

    # Display clean metrics in 3 columns
    col1, col2, col3 = st.columns(3)
    col1.metric("Days Observed", total_days)
    col2.metric("Expected Exceptions", f"{expected_failures:.1f}")
    col3.metric("Actual Exceptions", num_failures)

    try: 
        if num_failures == 0:
            lr_stat = -2 * np.log((expected_success_rate ** total_days))
        else:
            numerator = (expected_success_rate ** num_success) * (expected_failure_rate ** num_failures)
            denominator = (actual_success_rate ** num_success) * (actual_failure_rate ** num_failures)
            lr_stat = -2 * np.log(numerator / denominator)
        
        p_val = 1 - chi2.cdf(lr_stat, 1)

        st.write(f"**P-Value:** {p_val:.3f}")

        if p_val > 0.05: 
            st.success("✅ PASS: The model is accurate. Failures are within the expected random range.")
        else:
            if num_failures > expected_failures:
                st.error("⚠️ FAIL: The model UNDER-estimates risk. Losses happen too often.")
            else:
                st.warning("⚠️ FAIL: The model OVER-estimates risk. You are too safe.")
                
    except Exception as e:
        st.error(f"Could not calculate Kupiec Test: {e}")

# 1. Sidebar Inputs
symbol = st.sidebar.text_input("Ticker Symbol", value="NDX")
days_ahead = st.sidebar.slider("Days to Simulate", 30, 365, value=252)
stress_test = st.sidebar.checkbox("Activate Bear Market Stress Test?", value=False)

# 2. Get Data
data = yf.download(symbol, period="5y")['Close']
returns = np.log(data / data.shift(1)).dropna()
current_price = data.iloc[-1]
# 3. Logic: Run Simulation when button is clicked
if st.button("Run Simulation"):
    
    num_simulations = 1000
    position_size_usd = 10000
    
    with st.spinner("Running 2,000 parallel universes (Base + Stress)..."):
        
        # Fit the historical data once
        params = t.fit(returns)
        
        # ==========================================
        # MODEL 1: BASE CASE (MLE)
        # ==========================================
        df_base, loc_base, scale_base = params 
        if df_base < 3: df_base = 3 # Safety cap
            
        sim_base = t.rvs(df_base, loc=loc_base, scale=scale_base, size=(days_ahead, num_simulations))
        paths_base = current_price * np.exp(np.cumsum(sim_base, axis=0))
        paths_base = np.vstack([np.full((1, num_simulations), current_price), paths_base])
        
        pnl_base = ((paths_base[-1] / current_price) * position_size_usd) - position_size_usd
        var_base = np.sort(pnl_base)[int(num_simulations * 0.05)]

        # ==========================================
        # MODEL 2: STRESS TEST (Bear Market)
        # ==========================================
        df_stress = 5 
        loc_stress = -0.001             # Forced negative drift
        scale_stress = params[2] * 1.5  # Forced high volatility
        
        sim_stress = t.rvs(df_stress, loc=loc_stress, scale=scale_stress, size=(days_ahead, num_simulations))
        paths_stress = current_price * np.exp(np.cumsum(sim_stress, axis=0))
        paths_stress = np.vstack([np.full((1, num_simulations), current_price), paths_stress])
        
        pnl_stress = ((paths_stress[-1] / current_price) * position_size_usd) - position_size_usd
        var_stress = np.sort(pnl_stress)[int(num_simulations * 0.05)]

        # ==========================================
        # PLOTTING HELPER FUNCTION
        # ==========================================
        # We write this so we don't have to type the graph code twice!
        def create_plots(paths, pnl, var, title):
            fig, ax = plt.subplots(1, 2, figsize=(15, 6))
            
            # Left: Paths
            ax[0].plot(paths[:, :100], alpha=0.1, color='blue')
            ax[0].plot(paths[:,0], color="black", linewidth=1, label="Sample Path")
            ax[0].set_title(f"Monte Carlo Paths ({title})")
            ax[0].grid(True, alpha=0.3)
            
            # Right: Histogram
            ax[1].hist(pnl + position_size_usd, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
            ax[1].axvline(x=position_size_usd, color='green', linewidth=1, label='Break Even')
            ax[1].axvline(x=position_size_usd + var, color='red', linestyle='--', linewidth=2, label=f"VaR 95%: ${var:,.0f}")
            ax[1].set_title(f"Terminal Wealth ({title})")
            ax[1].set_ylabel("Number of Simulations")
            ax[1].legend()
            ax[1].grid(True, alpha=0.3)
            
            return fig

        # ==========================================
        # STREAMLIT UI RENDER (TABS)
        # ==========================================
        tab1, tab2 = st.tabs(["📈 Base Case (Historical)", "📉 Stress Test (Bear Market)"])
        
        with tab1:
            st.subheader("Scenario 1: Historical Market Fit")
            st.pyplot(create_plots(paths_base, pnl_base, var_base, "Base Case"))
            st.metric("Projected VaR (95%)", f"${var_base:,.2f}")
            
        with tab2:
            st.subheader("Scenario 2: Simulated Recession")
            st.info("Assumes a -25% annualized downward drift and a 50% spike in volatility.")
            st.pyplot(create_plots(paths_stress, pnl_stress, var_stress, "Stress Test"))
            st.metric("Stress Test VaR (95%)", f"${var_stress:,.2f}")

        # Run the Backtest below everything
        st.divider() 
        run_backtest_ui(returns)