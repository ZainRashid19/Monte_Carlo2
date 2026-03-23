import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t, norm, chi2

# ==========================================
# 1. PAGE CONFIG & BLOOMBERG UI OVERRIDE
# ==========================================
st.set_page_config(page_title="Monte Carlo Risk Engine", layout="wide")

st.markdown("""
    <style>
    /* Main background and font */
    .stApp {
        background-color: #050505;
        color: #39ff14; /* Neon Green */
        font-family: 'Courier New', Courier, monospace;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffb000 !important; /* Bloomberg Amber */
        text-transform: uppercase;
        border-bottom: 1px solid #ffb000;
        padding-bottom: 5px;
    }
    
    /* Metrics / KPIs */
    [data-testid="stMetricLabel"] {
        color: #00ffff !important; /* Cyan for labels */
        font-weight: bold;
    }
    [data-testid="stMetricValue"] {
        color: #39ff14 !important; /* Neon Green for numbers */
    }
    
    /* Alert Boxes (Kupiec Test) */
    .stAlert {
        background-color: #111111 !important;
        border: 1px solid #39ff14;
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

st.title("SYS >> MONTE CARLO RISK ENGINE [TERMINAL]")

# ==========================================
# 2. BACKTESTING FUNCTION
# ==========================================
def run_backtest_ui(returns, confidence_level=0.95):
    st.subheader(">> SYSTEM DIAGNOSTIC: KUPIEC POF TEST")
    
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

    col1, col2, col3 = st.columns(3)
    col1.metric("DAYS OBSERVED", total_days)
    col2.metric("EXPECTED EXCEPTIONS", f"{expected_failures:.1f}")
    col3.metric("ACTUAL EXCEPTIONS", num_failures)

    try: 
        if num_failures == 0:
            lr_stat = -2 * np.log((expected_success_rate ** total_days))
        else:
            numerator = (expected_success_rate ** num_success) * (expected_failure_rate ** num_failures)
            denominator = (actual_success_rate ** num_success) * (actual_failure_rate ** num_failures)
            lr_stat = -2 * np.log(numerator / denominator)
        
        p_val = 1 - chi2.cdf(lr_stat, 1)

        st.write(f"**P-VALUE:** {p_val:.3f}")

        if p_val > 0.05: 
            st.success("✅ PASS: The model is accurate. Failures are within the expected random range.")
        else:
            if num_failures > expected_failures:
                st.error("⚠️ FAIL: The model UNDER-estimates risk. Losses happen too often.")
            else:
                st.warning("⚠️ FAIL: The model OVER-estimates risk. You are too safe.")
                
    except Exception as e:
        st.error(f"Could not calculate Kupiec Test: {e}")

# ==========================================
# 3. COMMAND INPUTS 
# ==========================================
st.subheader(">> INPUT PARAMETERS")
cmd_col1, cmd_col2, cmd_col3 = st.columns(3)

with cmd_col1:
    symbol = st.text_input("TICKER SYMBOL", value="NDX")
with cmd_col2:
    days_ahead = st.number_input("DAYS TO SIMULATE", min_value=30, max_value=365, value=252)
with cmd_col3:
    st.write("") 
    st.write("")
    run_btn = st.button("EXECUTE SIMULATION >>")

# ==========================================
# 4. MAIN EXECUTION LOGIC
# ==========================================
if run_btn:
    num_simulations = 1000
    position_size_usd = 10000
    
    with st.spinner(f"FETCHING DATA AND RUNNING 2,000 PARALLEL UNIVERSES FOR {symbol}..."):
        
        # Fetch Data (with the Pandas Series fix)
        data = yf.download(symbol, period="5y")['Close']
        returns = np.log(data / data.shift(1)).dropna()
        current_price = float(np.array(data.iloc[-1]).item())
        
        params = t.fit(returns)
        
        # --- MODEL 1: BASE CASE (MLE) ---
        df_base, loc_base, scale_base = params 
        if df_base < 3: df_base = 3 
            
        sim_base = t.rvs(df_base, loc=loc_base, scale=scale_base, size=(days_ahead, num_simulations))
        paths_base = current_price * np.exp(np.cumsum(sim_base, axis=0))
        paths_base = np.vstack([np.full((1, num_simulations), current_price), paths_base])
        
        pnl_base = ((paths_base[-1] / current_price) * position_size_usd) - position_size_usd
        var_base = np.sort(pnl_base)[int(num_simulations * 0.05)]

        # --- MODEL 2: STRESS TEST (Bear Market) ---
        df_stress = 5 
        loc_stress = -0.001            
        scale_stress = params[2] * 1.5 
        
        sim_stress = t.rvs(df_stress, loc=loc_stress, scale=scale_stress, size=(days_ahead, num_simulations))
        paths_stress = current_price * np.exp(np.cumsum(sim_stress, axis=0))
        paths_stress = np.vstack([np.full((1, num_simulations), current_price), paths_stress])
        
        pnl_stress = ((paths_stress[-1] / current_price) * position_size_usd) - position_size_usd
        var_stress = np.sort(pnl_stress)[int(num_simulations * 0.05)]

        # --- PLOTTING HELPER FUNCTION (DARK MODE) ---
        def create_plots(paths, pnl, var, title):
            plt.style.use('dark_background')
            fig, ax = plt.subplots(1, 2, figsize=(15, 6), facecolor='#050505')
            
            # Left: Paths
            ax[0].set_facecolor('#050505')
            ax[0].plot(paths[:, :100], alpha=0.1, color='#00ffff')
            ax[0].plot(paths[:,0], color="#ff0000", linewidth=1.5, label="Sample Path")
            ax[0].set_title(f"PRICE TRAJECTORY ({title})", color='#ffb000')
            ax[0].grid(True, color='#333333', linestyle=':')
            
            # Right: Histogram
            ax[1].set_facecolor('#050505')
            ax[1].hist(pnl + position_size_usd, bins=50, color='#0055ff', edgecolor='#00aaff', alpha=0.7)
            ax[1].axvline(x=position_size_usd, color='#39ff14', linewidth=2, label='BREAK EVEN')
            ax[1].axvline(x=position_size_usd + var, color='#ff003c', linestyle='--', linewidth=2, label=f"VaR 95%: ${var:,.0f}")
            ax[1].set_title(f"TERMINAL WEALTH DIST ({title})", color='#ffb000')
            ax[1].legend(facecolor='#111111', edgecolor='#ffb000', labelcolor='white')
            ax[1].grid(True, color='#333333', linestyle=':')
            
            return fig

        # --- UI RENDER (TABS) ---
        st.write("") # Spacing
        tab1, tab2 = st.tabs(["📈 BASE CASE (HISTORICAL)", "📉 STRESS TEST (BEAR MARKET)"])
        
        with tab1:
            st.subheader(">> SCENARIO 1: HISTORICAL MARKET FIT")
            st.pyplot(create_plots(paths_base, pnl_base, var_base, "BASE CASE"))
            st.metric("PROJECTED VaR (95%)", f"${var_base:,.2f}")
            
        with tab2:
            st.subheader(">> SCENARIO 2: SIMULATED RECESSION")
            st.info("SYSTEM ASSUMES A -25% ANNUALIZED DOWNWARD DRIFT AND A 50% SPIKE IN VOLATILITY.")
            st.pyplot(create_plots(paths_stress, pnl_stress, var_stress, "STRESS TEST"))
            st.metric("STRESS TEST VaR (95%)", f"${var_stress:,.2f}")

        # --- BACKTEST RENDER ---
        st.divider() 
        run_backtest_ui(returns)