
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime
from scipy.stats import chi2
# for monte
import pandas as pd
from scipy.stats import t



# MonteCarlo sim
def monte_carlo_simulation(returns, current_price, position_size_usd, days_ahead=252, num_simulations=1000):
    print("\n" + "="*50)
    print(f"Monte Carlo Simulation {days_ahead} Days Ahead")
    print(f"Model: Student's t-distribution (Fat Tail Adjusted)")
    print("="*50)


    # -- Stress test activation part  ---
    STRESS_TEST_MODE = False  # <--- Change to False to see the "Normal" market
    params = t.fit(returns)
    
    if STRESS_TEST_MODE:
        print(f"!!! STRESS TEST ACTIVE: BEAR MARKET SCENARIO !!!")
        df = 5  # bad market df 
        loc = -0.001         #-0.001 for negative drift, drops by %0.1 everyday 
        scale = params[2] * 1.5 # Higher volatility by 50%
    else:
        print(f"--- Standard Market Model (Historical Data) ---")
        df, loc, scale = params 
        # if the df around 2smth 
        if df < 3: #inside the else statement 
            print("   (Note: Capping extreme tail risk at df=3 for stability)") 
            df = 3

    # The rest of the code stays exactly the same...
    simulated_returns = t.rvs(df, loc=loc, scale=scale, size=(days_ahead, num_simulations))
    cumulative_returns = np.cumsum(simulated_returns, axis=0)
    price_paths = current_price * np.exp(cumulative_returns) # Eulers number computation w np.exp

    start_row = np.full((1,num_simulations), current_price)
    price_paths = np.vstack([start_row, price_paths])

    final_prices = price_paths[-1]
    final_values = (final_prices / current_price * position_size_usd)
    profit_loss = final_values - position_size_usd

    cutoff_index = int(num_simulations *0.05) # for bottom 5%
    sorted_pl = np.sort(profit_loss)
    mc_var_95 = sorted_pl[cutoff_index]

    print(f"Projected bottom 5% Outcome (1 Year): {mc_var_95:,.2f}")

    plt.figure(figsize=(14,7))

    plt.subplot(1,2,1)
    plt.plot(price_paths[:, :100], alpha=0.1, color="blue")
    plt.plot(price_paths[:,0], color="black", linewidth=1, label="Sample Path") # highlighted 
    plt.title(f"Monte Carlo Paths (Next {days_ahead} Days)")
    plt.xlabel("Days")
    plt.ylabel("Price ($)")
    plt.grid(True, alpha = 0.3)

    plt.subplot(1,2,2)
    plt.hist(final_values, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
    plt.ylabel("Number of Simulations")
    plt.axvline(x=position_size_usd, color='green', linewidth=1, label="Break Even")
    plt.title("Distribution of Portfolio Value (Year End)")
    plt.xlabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)


    plt.tight_layout()
    plt.show()


    # Benchmark comprassion portion 
    mu = np.mean(returns)
    sigma = np.std(returns)

    normal_var_percent = (mu * days_ahead) - (1.645 * sigma *np.sqrt(days_ahead))

    normal_end_price = current_price * (1+normal_var_percent)
    normal_pnl =(normal_end_price / current_price * position_size_usd) - position_size_usd


    print("\n" + "="*40)
    print("MODEL COMPARISON (Why Fat Tails Matter)")
    print(f"Standard (Normal) VaR Limit: ${normal_pnl:,.2f}")
    print(f"Your (Fat Tail) VaR Limit:   ${mc_var_95:,.2f}")
    print(f"Risk Underestimation:        ${(normal_pnl - mc_var_95):,.2f}")
    print("="*40 + "\n")

#plotting graphs :)

def plot_graph(returns,var_cutoff_percent,symbol):
    plt.figure(figsize=(12,6))

    #daily
    pos_returns = returns[returns>0]
    neg_returns= returns[returns<=0]

    plt.bar(pos_returns.index, pos_returns,color="green", alpha=0.5, label="Up days")
    plt.bar(neg_returns.index, neg_returns,color="red", alpha=0.5, label="Down days")

    #var line
    plt.axhline(y=-var_cutoff_percent, color="darkred", linestyle = "--", linewidth=2, label=f"VaR limit({-var_cutoff_percent:.1%})")

    #highlighting the failures
    failures = returns[returns<-var_cutoff_percent]
    plt.scatter(failures.index, failures,color="Black", edgecolors="Black", s=60, zorder=5, label="Failures")

    #colors
    plt.title("Visualizing Risk: {symbol} vs. The limit", fontsize=15, fontweight="bold")
    plt.ylabel("Daily Return (%)")
    plt.xlabel("Time frame")
    plt.axhline(y=0,color="black", linewidth=0.5) # zero line 
    plt.legend(loc='upper left', frameon=True, fancybox=True, framealpha=0.9)
    plt.grid(True, alpha=0.3)

    plt.show()



# BACKTESTING
def backtest(returns,position_size, confidence_level=0.95):
    print("-" * 30)
    print("BACKTESTING")
    print("-"*30)

    z_score = norm.ppf(confidence_level)
    daily_vol = returns.std()
    var_cuttoff_percent = daily_vol * z_score

    actual_failures = returns[returns< -var_cuttoff_percent]
    num_failures = len(actual_failures)
    total_days = len(returns)
    num_success = total_days - num_failures

    expected_failure_rate = 1- confidence_level
    expected_success_rate = confidence_level

    actual_failure_rate = num_failures/total_days
    actual_success_rate = 1- actual_failure_rate
    expected_failures = total_days * expected_failure_rate

    print(f"Days Observed:            {total_days}")
    print(f"Expected Exceptions:      {expected_failures:.1f} days ({expected_failure_rate:.1%})")
    print(f"Actual Exceptions:        {num_failures} days ({actual_failure_rate:.1%})")

    try: 
        #lr = likelood ratio
        if num_failures == 0:
            lr_stat = -2 * np.log(( expected_success_rate** total_days))
        
        else:
            #Probability of seeing this result if the Model is CORRECT
            numerator = (expected_success_rate**num_success) * (expected_failure_rate**num_failures)
            #Probability of seeing this result if the Reality is CORRECT
            denominator = (actual_success_rate**num_success) * (actual_failure_rate ** num_failures)
            lr_stat= -2*np.log(numerator/denominator)
        
        #critical val (chi sq distribution w 1 degree of freedom, still 95% confidence interval) 
        # cdf = cumlulative distribution function 
        p_val = 1 - chi2.cdf(lr_stat,1)

        print("-" * 30)
        print(f"VERDICT (P-val:{p_val:.3f})")

        if p_val>0.05: 
            # pass 
            print(f" \u2705 PASS: The model is accurate.")
            print(f"   (Failures are within the expected random range).")
        else:
            # warning sign 
            print(f" \u26A0 FAIL: The model is inaccurate.")
            if(num_failures>expected_failures):
                print("   (It UNDER-estimates risk. Losses happen too often).")
            else:
                print("   (It OVER-estimates risk. You are too safe).")
        
        
    except Exception as e:
        print(f"Could not calculate Kupiec Test: {e}")
    
    print("="*50)

# assumtion of 10k stock
def calculate_market_risk(symbol, position_size_usd=10000):
    try:
        stock = yf.Ticker(symbol)

        history = stock.history(period="1y")

        if history.empty:
            print (f"No data found for symbol: {symbol}")
            return
        

        #Volatility formula
        history['Log_Returns'] = np.log(history['Close']/history['Close'].shift(1))
        returns = history['Log_Returns'].dropna()

        daily_vol = returns.std()
        weekly_vol = daily_vol * np.sqrt(5)
        annual_vol = daily_vol * np.sqrt(252)
        # 252 is std # of trading days in the US Stock market

        # calculating the neg returns of the stock
        negative_returns = returns.copy()
        #ignores pos days 
        negative_returns[negative_returns>0]=0
        daily_downside_vol = np.sqrt(np.mean(negative_returns**2))
        weekly_downside_vol = daily_downside_vol * np.sqrt(5)
        annual_downside_vol = daily_downside_vol * np.sqrt(252)


        # calculating the pos returns of the stock
        positive_returns = returns.copy()
        #ignore the neg days 
        positive_returns[positive_returns<0]=0
        daily_upside_vol = np.sqrt(np.mean(positive_returns**2))
        weekly_upside_vol = daily_upside_vol * np.sqrt(5)
        annual_upside_vol = daily_upside_vol * np.sqrt(252)

        #Sortino ratio 

        avg_daily_returns = returns.mean()
        annual_returns = avg_daily_returns*252
        sortino_ratio = annual_returns/annual_downside_vol
        
        #95% confidence or 1.645 one tailed  z score 
        confidence_level = 0.95 
        z_score = norm.ppf(confidence_level)
        
        #VaR's
        one_day_var_percent = daily_vol * z_score
        one_day_var_dollar = position_size_usd * one_day_var_percent

        one_day_var_percent_gain = daily_upside_vol * z_score
        one_day_var_dollar_gain = position_size_usd * one_day_var_percent_gain 

        # -1 is the most recent data 
        current_price = history['Close'].iloc[-1]
        print("=" * 50)

        print(f"RISK ANALYSIS: {symbol.upper()}")
        print(f"Position Size: ${position_size_usd:,.2f}")
        print("="*50)

        print(f"Current Price: ${current_price:.2f}")
        print("-"*30)

        print(f"1: TOTAL VOLATILITY")
        print(f"Daily volatility: {daily_vol:.2%}")
        print(f"Weekly volatility: {weekly_vol:.2%}")
        print(f"Yearly Volatility: {annual_vol:.2%}")

        print("-"*30)
        
        print(f"2. DOWNSIDE VOLATILITY (Only Negative Days)")
        print(f"Daily:   {daily_downside_vol:.2%}")
        print(f"Weekly:  {weekly_downside_vol:.2%}")
        print(f"Annual:  {annual_downside_vol:.2%}")

        print("-"*30)

        # Sortino ratio print statements 

        print(f"SORTINO RATIO:{sortino_ratio:.2f}")
        if sortino_ratio>2:
            print("\2705 EXCELENT: High returns for low bad risk")
        elif sortino_ratio>1:
            print("\u2705 GOOD: You are getting paid for your risk")
        else:
            print("\u26A0 CAUTION: Low returns compared to the downside risk")

        print("-"*30)



        # Debug block for more accurate sortino calculation
        # # ... (Your existing Sortino calculation) ...
        # sortino_ratio = annual_returns / annual_downside_vol
        
        # # --- ADD THIS DEBUG BLOCK ---
        # print("\n--- DEBUGGING SORTINO ---")
        # print(f"Annualized Return:       {annual_returns:.2%}")
        # print(f"Annual Downside Risk:    {annual_downside_vol:.2%}")
        # print(f"Risk-Free Adjustment:    {(annual_returns - 0.045):.2%} (Assuming 4.5% rate)")
        # adjusted_sortino = (annual_returns - 0.045) / annual_downside_vol
        # print(f"Adjusted Sortino (w/ RF): {adjusted_sortino:.4f}") 
        # print("-" * 30)
        

        print(f"3. Positive VOLATILITY (Only Positive Days)")
        print(f"Daily:   {daily_upside_vol:.2%}")
        print(f"Weekly:  {weekly_upside_vol:.2%}")
        print(f"Annual:  {annual_upside_vol:.2%}")

        print("-"*30)


        # Comparison Logic
        if annual_downside_vol < annual_vol:
            print("Good News: 'Bad' volatility is lower than total volatility.")
            print("(The stock surges up more violently than it crashes down).")
        else:
            print("Warning: 'Bad' volatility is higher than total volatility.")
            print("(The stock crashes harder than it rallies).")
        print("-" * 30)

        #Max losses based of 95% confidence interval
        cutoff = -one_day_var_percent

        worst_days=returns[returns<cutoff]

        if len(worst_days)>0:
            cvar_percent = worst_days.mean()
            cvar_dollar = position_size_usd*cvar_percent
        else:
            # Fallback if history was unusually calm
            pdf_at_z = norm.pdf(z_score)       # This is phi(z)
            cdf_at_z = norm.cdf(-z_score)      # This is (1-alpha) or 0.05
            # This is the formula: sigma * (pdf / cdf)
            cvar_percent = -daily_vol * (pdf_at_z / cdf_at_z)
            cvar_dollar = position_size_usd * cvar_percent
        
        print(f"4: RISK SENARIOS (95% CONFIDENCE)")
        print("-"*50)
        print(f"A: The normal worst case (VaR):")
        print(f" You are 95% safe. But if you hit the bottom 5%")
        print(f" Limit Loss: ${one_day_var_dollar:,.2f} {one_day_var_percent:.2%}")
        print(f"")
        print(f"B: The True Disaster(CVaR/Expected Shortfall):")
        print(f" IF the line breaks, this is an AVERAGE CRASH PRICE")
        print(f" Expected Crash: ${(cvar_dollar):,.2f} ({abs(cvar_percent):.2%})")

        print("-"*30)

        ratio = abs(cvar_dollar/one_day_var_dollar)

        # perfect ratio is 1.3 rounded, (small overestimate as compared to 1.254)
        if (ratio>1.3):
            print(" \u26A0: FAT TAIL WARNING: Huge Crash risk")
            print("When this stock crashes, it fails CATASTROPHICALLY")
            print (" (Real loss is 30% larger than expected loss)")
        else:
            print("\u2705 Normal Risk: Crashes are predictable")
            print("(The crash is close to the predicition)")
        
        print("-"*50)


        backtest(returns, position_size_usd, confidence_level)

        plot_graph(returns,one_day_var_percent,symbol)

        monte_carlo_simulation(returns,current_price, position_size_usd)

    except Exception as e:
        print(f"Error calculating risk for {symbol}: {e}")


def main():
    symbol = "NFLX"
    my_investment = 81
    calculate_market_risk(symbol,my_investment)

if __name__ == "__main__":
    main()

