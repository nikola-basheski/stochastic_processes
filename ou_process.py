import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
#%%


# Import dependencies
import time
import math
import numpy as np
import pandas as pd
import datetime
import scipy as sc
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from IPython.display import display, Latex
from statsmodels.graphics.tsaplots import plot_acf


#%%

ARI = pd.read_csv(r"/home/nikola/Pictures/ARI.csv")



FBRT = pd.read_csv(r"/home/nikola/Pictures/BXMT (1).csv")

#%%

#ARI = ARI.iloc[-557:, :]


#%%

pearson_corr = ARI['Open'].corr(FBRT['Open'], method='spearman')

#%%
Ari_sample = pd.Series(list(ARI['Close']))#.iloc[-len(ARI):-780]))

Fbrt_sample = pd.Series(list(FBRT['Close']))#.iloc[-len(ARI):-780]))


#%%

pearson_corr = Ari_sample.corr(Fbrt_sample, method='spearman')

print(pearson_corr)


#%%
data1 = Ari_sample.copy()
data2 = Fbrt_sample.copy()

#%%


df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# Reset indices
df1_reset = df1.reset_index(drop=True)
df2_reset = df2.reset_index(drop=True)

# Specify the correlation threshold
correlation_threshold = 0

# Set the window size for the rolling correlation
window_size = 40  # Adjust as needed

# Create a DataFrame to store the results
result_df = pd.DataFrame(columns=['Start Index', 'End Index', 'Correlation'])

# Iterate through the rolling window
for i in range(len(df1_reset) - window_size + 1):
    df1_window = Ari_sample.iloc[i:i + window_size]#.values
    df2_window = Fbrt_sample.iloc[i:i + window_size]#.values
    current_correlation = df1_window.corr(df2_window, method='pearson')
    #print(df1_window)
    print(current_correlation)
    # Check if the correlation falls below the threshold
    if current_correlation < correlation_threshold:
        result_df = result_df.append({
            'Start Index': i,
            'End Index': i + window_size - 1,
            'Correlation': current_correlation
        }, ignore_index=True)

# Display the result DataFrame
print(result_df)


#%%


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


#%%
x1  = Ari_sample
x2 = Fbrt_sample

# Create a DataFrame
df = pd.DataFrame({'x1': x1, 'x2': x2})

# Calculate rolling correlation with a window size of 30
rolling_corr = df['x1'].rolling(window=40).corr(df['x2'])

# Find the index where the correlation is below 0.5
breakpoints = rolling_corr[rolling_corr < 0].index

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(df['x1'], label='ARI', linewidth=2)
plt.plot(df['x2'], label='BMXT', linewidth=2)

# Highlight regions with low correlation
for i in range(0, len(breakpoints)-1, 2):
    plt.axvspan(breakpoints[i], breakpoints[i + 1], color='yellow', alpha=0.3)

# Add labels and legend
plt.xlabel('Days from January 1st, 2019')
plt.ylabel('Price')
plt.title('Overlay of BXMT and ARI Prices with Low Correlation Highlighted')
plt.legend()
plt.show()

#%%
# Assuming you have x1 and x2 defined
x1 = Ari_sample.copy()
x2 = Fbrt_sample.copy()

# Calculate the spread
spread = x1 - x2
# Calculate the ratio
ratio = x1 / x2

# Calculate the 40-day moving average
moving_average = ratio.rolling(window=40).mean()

k0 = moving_average.fillna(np.mean(moving_average.fillna(0)))
# Define the O-U process
def ornstein_uhlenbeck(theta, x):
    kappa, theta_ou, sigma = theta
    dt = 1  # Assuming daily data

    ou_process = np.zeros_like(x)
    ou_process[0] = x.iloc[0]

    for t in range(1, len(x)):
        dW = np.random.normal(0, np.sqrt(dt))
        ou_process[t] = (
            ou_process[t - 1] * np.exp(-kappa * dt)
            + theta_ou * (1 - np.exp(-kappa * dt))
            + sigma * np.sqrt((1 - np.exp(-2 * kappa * dt)) / (2 * kappa)) * dW
        )

    return ou_process

# Define the log-likelihood function for O-U process
def log_likelihood_OU(theta, x):
    ou_process = ornstein_uhlenbeck(theta, x)
    residuals = x - ou_process
    sigma2 = np.var(residuals)
    log_likelihood_val = -len(x) * np.log(np.sqrt(2 * np.pi * sigma2)) - np.sum(residuals ** 2) / (2 * sigma2)
    return -log_likelihood_val

# Set up optimization
initial_guess = [1, np.mean(spread), np.std(spread)]
bounds = [(0, None), (None, None), (0, None)]  # Kappa and Sigma must be positive

# Perform optimization
result = minimize(log_likelihood_OU, initial_guess, args=(spread,), bounds=bounds)

# Extract optimized parameters
kappa_opt, theta_opt, sigma_opt = result.x

# Display the optimized parameters
print('Optimized Parameters:')
print(f'Kappa: {kappa_opt}')
print(f'Theta: {theta_opt}')
print(f'Sigma: {sigma_opt}')

# Simulate O-U process with optimized parameters
simulated_ou_process = ornstein_uhlenbeck([kappa_opt, theta_opt, sigma_opt/10], spread)

# Generate buy/sell signals
buy_signal = np.where(spread < simulated_ou_process, 1, 0)
sell_signal = np.where(spread > simulated_ou_process, -1, 0)

rolling_corr = x1.rolling(window=40).corr(x2)

# Find the index where the correlation is below 0
correlation_breakpoints = rolling_corr[rolling_corr < 0].index



# Print correlation breakpoints for debugging
print("Correlation Breakpoints:", correlation_breakpoints)

# Plot the actual spread, the simulated O-U process, buy/sell signals, correlation breakpoints, and stock prices
plt.figure(figsize=(10, 6))
plt.plot(spread, label='Actual Spread', linewidth=2)
plt.plot(simulated_ou_process, label='Simulated O-U Process', linestyle='--', linewidth=2)
plt.scatter(np.where(buy_signal == 1)[0], spread[buy_signal == 1], marker='^', color='g', label='Buy Signal')
plt.scatter(np.where(sell_signal == -1)[0], spread[sell_signal == -1], marker='v', color='r', label='Sell Signal')

# Check if correlation breakpoints exist before plotting
if len(correlation_breakpoints) > 0:
    plt.scatter(correlation_breakpoints, spread.iloc[correlation_breakpoints], marker='o', color='y', label='Correlation Breakpoint')
    plt.axvline(x=correlation_breakpoints[0], color='yellow', linestyle='--', label='Correlation Turns Negative')
    # Extend the yellow line across the entire plot
    plt.axvline(x=correlation_breakpoints[0], color='yellow', linestyle='--')  

# Adjust the x-axis limits to zoom in on the correlation break
plt.xlim(correlation_breakpoints[0] - 10, correlation_breakpoints[0] + 10)

plt.xlabel('Time')
plt.ylabel('Spread')
plt.title('Actual Spread vs. Simulated O-U Process with Buy/Sell Signals and Correlation Breakpoints')
plt.legend()
plt.show()
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Assuming you have x1 and x2 defined
x1 = Ari_sample.copy()
x2 = Fbrt_sample.copy()


# Date range from June 1, 2019, to June 1, 2024
date_range = pd.date_range(start='2019-06-01', end='2024-06-01', freq='D')


k0 = np.mean(x1/x2)

# Calculate the ratio
ratio = x1 / x2

# Calculate the 40-day moving average
moving_average = ratio.rolling(window=40).mean()

k0 = moving_average.fillna(np.mean(moving_average.fillna(0)))
# Calculate the spread
spread = x1/(k0*x2)

# Calculate the spread
spread = x1/(k0*x2)
#spread = x1 - x2
# Define the O-U process
def ornstein_uhlenbeck(theta, x):
    kappa, theta_ou, sigma = theta
    dt = 1  # Assuming daily data

    ou_process = np.zeros_like(x)
    ou_process[0] = x.iloc[0]

    for t in range(1, len(x)):
        dW = np.random.normal(0, np.sqrt(dt))
        ou_process[t] = (
            ou_process[t - 1] * np.exp(-kappa * dt)
            + theta_ou * (1 - np.exp(-kappa * dt))
            + sigma * np.sqrt((1 - np.exp(-2 * kappa * dt)) / (2 * kappa)) * dW
        )

    return ou_process

# Define the log-likelihood function for O-U process
def log_likelihood_OU(theta, x):
    ou_process = ornstein_uhlenbeck(theta, x)
    residuals = x - ou_process
    sigma2 = np.var(residuals)
    log_likelihood_val = -len(x) * np.log(np.sqrt(2 * np.pi * sigma2)) - np.sum(residuals ** 2) / (2 * sigma2)
    return -log_likelihood_val

# Set up optimization
initial_guess = [1, np.mean(spread), np.std(spread)]
bounds = [(0, None), (None, None), (0, None)]  # Kappa and Sigma must be positive

# Perform optimization
result = minimize(log_likelihood_OU, initial_guess, args=(spread,), bounds=bounds)

# Extract optimized parameters
kappa_opt, theta_opt, sigma_opt = result.x

# Display the optimized parameters
print('Optimized Parameters:')
print(f'Kappa: {kappa_opt}')
print(f'Theta: {theta_opt}')
print(f'Sigma: {sigma_opt}')

# Simulate O-U process with optimized parameters
simulated_ou_process = ornstein_uhlenbeck([kappa_opt, theta_opt, sigma_opt], spread)

# Generate buy/sell signals
buy_signal = np.where(spread < simulated_ou_process, 1, 0)
sell_signal = np.where(spread > simulated_ou_process, -1, 0)

# Find the index where correlation turns negative
rolling_corr = x1.rolling(window=40).corr(x2)

# Find the index where the correlation is below 0
correlation_breakpoints = rolling_corr[rolling_corr < 0].index


# Plot the actual spread, the simulated O-U process, buy/sell signals, correlation breakpoints, mean spread, and stock prices
plt.figure(figsize=(10, 6))
plt.plot(spread, label='Actual Spread', linewidth=2)
plt.plot(simulated_ou_process, label='Simulated O-U Process', linestyle='--', linewidth=2)
plt.scatter(np.where(buy_signal == 1)[0], spread[buy_signal == 1], marker='^', color='g', label='Buy Signal')
plt.scatter(np.where(sell_signal == -1)[0], spread[sell_signal == -1], marker='v', color='r', label='Sell Signal')
if len(correlation_breakpoints) > 0:
    plt.scatter(correlation_breakpoints, spread.iloc[correlation_breakpoints], marker='o', color='y', label='Correlation Breakpoint')
    plt.axvline(x=correlation_breakpoints[0], color='yellow', linestyle='--', label='Correlation Turns Negative')
plt.axvline(x=correlation_breakpoints[0], color='yellow', linestyle='--')  # Extend the yellow line across the entire plot

# Plot the mean spread
mean_spread = np.mean(spread)
plt.axhline(y=mean_spread, color='orange', linestyle='-', label='Mean Spread')

# Adjust the x-axis limits to zoom in on the correlation break
plt.xlim(correlation_breakpoints[0] - 500, correlation_breakpoints[0] + 500)

plt.xlabel('Time')
plt.ylabel('Ratio')
plt.title('Actual Ratio vs. Simulated O-U Process with Buy/Sell Signals and Correlation Breakpoints')
plt.legend()
plt.show()
#%%
# Assuming buy_signal and sell_signal are already defined
initial_investment = 100000
portfolio_value = initial_investment
cash = initial_investment  # Initially, the entire portfolio is in cash
stock_value_x1 = 0  # Value of the position in the first stock (Ari_sample)
stock_value_x2 = 0  # Value of the position in the second stock (Fbrt_sample)

# Iterate through the signals
for i in range(len(buy_signal)):
    if buy_signal[i] == 1:
        # Buy signal: Invest half of the portfolio value in each stock
        stock_price_x1 = Ari_sample.iloc[i]  # Replace with the actual price of the first stock at index i
        stock_price_x2 = Fbrt_sample.iloc[i]  # Replace with the actual price of the second stock at index i

        investment_amount = 0.005 * cash

        shares_x1 = investment_amount / stock_price_x1
        shares_x2 = investment_amount / stock_price_x2

        stock_value_x1 += shares_x1 * stock_price_x1
        stock_value_x2 += shares_x2 * stock_price_x2

        cash -= investment_amount
    elif sell_signal[i] == -1:
        # Sell signal: Convert the entire portfolio value to cash
        stock_price_x1 = Ari_sample.iloc[i]  # Replace with the actual price of the first stock at index i
        stock_price_x2 = Fbrt_sample.iloc[i]  # Replace with the actual price of the second stock at index i

        cash += stock_value_x1 + stock_value_x2

        stock_value_x1 = 0
        stock_value_x2 = 0

# Calculate the final portfolio value
final_portfolio_value = cash + stock_value_x1 + stock_value_x2

# Calculate the profit
profit = final_portfolio_value - initial_investment

print(f"Initial Investment: ${initial_investment:.2f}")
print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
print(f"Profit: ${profit:.2f}")



#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Assuming you have x1 and x2 defined
x1 = Ari_sample.copy()
x2 = Fbrt_sample.copy()

# Calculate the spread
# Assuming you have x1 and x2 defined
x1 = Ari_sample.copy()
x2 = Fbrt_sample.copy()

k0 = np.mean(x1/x2)
# Calculate the ratio
ratio = x1 / x2

# Calculate the 40-day moving average
moving_average = ratio.rolling(window=40).mean()

k0 = moving_average.fillna(np.mean(moving_average.fillna(0)))
# Calculate the spread
spread = x1/(k0*x2)


# Find the index where correlation turns negative
rolling_corr = x1.rolling(window=40).corr(x2)

# Find the index where the correlation is below 0
correlation_breakpoints = rolling_corr[rolling_corr < 0].index

# Zoom in on the relevant time period
start_index = max(correlation_breakpoints[0] - 40, 0)
end_index = min(correlation_breakpoints[0] + 40, len(x1))

# Plot the stock prices with zoomed-in view
plt.figure(figsize=(10, 6))
plt.plot(x1[start_index:end_index], label='Stock 1', linewidth=2)
plt.plot(x2[start_index:end_index], label='Stock 2', linewidth=2)
plt.axvline(x=correlation_breakpoints[0], color='yellow', linestyle='--', label='Correlation Turns Negative')

plt.xlabel('Time')
plt.ylabel('Stock Prices')
plt.title('Stock Prices with Zoomed-In View at Correlation Break')
plt.legend()
plt.show()
#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Assuming you have x1 and x2 defined
x1 = Ari_sample.copy()
x2 = Fbrt_sample.copy()

# Calculate the spread
# Assuming you have x1 and x2 defined
x1 = Ari_sample.copy()
x2 = Fbrt_sample.copy()

k0 = np.mean(x1/x2)

# Calculate the ratio
ratio = x1 / x2

# Calculate the 40-day moving average
moving_average = ratio.rolling(window=40).mean()

k0 = moving_average.fillna(np.mean(moving_average.fillna(0)))
# Calculate the spread
spread = x1/(k0*x2)
#spread = x1 - x2
# Define the O-U process
def ornstein_uhlenbeck(theta, x):
    kappa, theta_ou, sigma = theta
    dt = 1  # Assuming daily data

    ou_process = np.zeros_like(x)
    ou_process[0] = x.iloc[0]

    for t in range(1, len(x)):
        dW = np.random.normal(0, np.sqrt(dt))
        ou_process[t] = (
            ou_process[t - 1] * np.exp(-kappa * dt)
            + theta_ou * (1 - np.exp(-kappa * dt))
            + sigma * np.sqrt((1 - np.exp(-2 * kappa * dt)) / (2 * kappa)) * dW
        )

    return ou_process

# Define the log-likelihood function for O-U process
def log_likelihood_OU(theta, x):
    ou_process = ornstein_uhlenbeck(theta, x)
    residuals = x - ou_process
    sigma2 = np.var(residuals)
    log_likelihood_val = -len(x) * np.log(np.sqrt(2 * np.pi * sigma2)) - np.sum(residuals ** 2) / (2 * sigma2)
    return -log_likelihood_val

# Set up optimization
initial_guess = [1, np.mean(spread), np.std(spread)]
bounds = [(0, None), (None, None), (0, None)]  # Kappa and Sigma must be positive

# Perform optimization
result = minimize(log_likelihood_OU, initial_guess, args=(spread,), bounds=bounds)

# Extract optimized parameters
kappa_opt, theta_opt, sigma_opt = result.x

# Display the optimized parameters
print('Optimized Parameters:')
print(f'Kappa: {kappa_opt}')
print(f'Theta: {theta_opt}')
print(f'Sigma: {sigma_opt}')

# Simulate O-U process with optimized parameters
simulated_ou_process = ornstein_uhlenbeck([kappa_opt, theta_opt, sigma_opt], spread)

# Generate buy/sell signals for each stock based on the O-U process simulation
buy_signal_x1 = np.where(x1 < simulated_ou_process, 1, 0)
sell_signal_x1 = np.where(x1 > simulated_ou_process, -1, 0)

buy_signal_x2 = np.where(x2 < simulated_ou_process, 1, 0)
sell_signal_x2 = np.where(x2 > simulated_ou_process, -1, 0)

# Plot the actual stock prices, O-U process simulations, and buy/sell signals
plt.figure(figsize=(12, 8))

# Plot Stock 1
plt.subplot(2, 1, 1)
plt.plot(x1, label='Actual ARI Price', linewidth=2)
plt.plot(simulated_ou_process, label='Simulated O-U Process for ARI', linestyle='--', linewidth=2)
plt.scatter(np.where(buy_signal_x1 == 1)[0], x1[buy_signal_x1 == 1], marker='^', color='g', label='Buy Signal')
plt.scatter(np.where(sell_signal_x1 == -1)[0], x1[sell_signal_x1 == -1], marker='v', color='r', label='Sell Signal')
plt.xlabel('Time')
plt.ylabel('ARI Price')
plt.title('ARI Price and Simulated O-U Process with Buy/Sell Signals')
plt.legend()

# Plot Stock 2
plt.subplot(2, 1, 2)
plt.plot(x2, label='Actual BXMT Price', linewidth=2)
plt.plot(simulated_ou_process, label='Simulated O-U Process for ', linestyle='--', linewidth=2)
plt.scatter(np.where(buy_signal_x2 == 1)[0], x2[buy_signal_x2 == 1], marker='^', color='g', label='Buy Signal')
plt.scatter(np.where(sell_signal_x2 == -1)[0], x2[sell_signal_x2 == -1], marker='v', color='r', label='Sell Signal')
plt.xlabel('Time')
plt.ylabel('BXMT Price')
plt.title('BXMT Price and Simulated O-U Process with Buy/Sell Signals')
plt.legend()

plt.tight_layout()
plt.show()
#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Assuming you have x1 and x2 defined
x1 = Ari_sample.copy()
x2 = Fbrt_sample.copy()

k0 = np.mean(x1/x2)

# Calculate the ratio
ratio = x1 / x2

# Calculate the 40-day moving average
moving_average = ratio.rolling(window=40).mean()

k0 = moving_average.fillna(np.mean(moving_average.fillna(0)))
# Calculate the spread
spread = x1/(k0*x2)

# Define the O-U process
def ornstein_uhlenbeck(theta, x):
    kappa, theta_ou, sigma = theta
    dt = 1  # Assuming daily data

    ou_process = np.zeros_like(x)
    ou_process[0] = x.iloc[0]

    for t in range(1, len(x)):
        dW = np.random.normal(0, np.sqrt(dt))
        ou_process[t] = (
            ou_process[t - 1] * np.exp(-kappa * dt)
            + theta_ou * (1 - np.exp(-kappa * dt))
            + sigma * np.sqrt((1 - np.exp(-2 * kappa * dt)) / (2 * kappa)) * dW
        )

    return ou_process

# Define the log-likelihood function for O-U process
def log_likelihood_OU(theta, x):
    ou_process = ornstein_uhlenbeck(theta, x)
    residuals = x - ou_process
    sigma2 = np.var(residuals)
    log_likelihood_val = -len(x) * np.log(np.sqrt(2 * np.pi * sigma2)) - np.sum(residuals ** 2) / (2 * sigma2)
    return -log_likelihood_val

# Set up optimization
initial_guess = [1, np.mean(spread), np.std(spread)]
bounds = [(0, None), (None, None), (0, None)]  # Kappa and Sigma must be positive

# Perform optimization
result = minimize(log_likelihood_OU, initial_guess, args=(spread,), bounds=bounds)

# Extract optimized parameters
kappa_opt, theta_opt, sigma_opt = result.x

# Display the optimized parameters
print('Optimized Parameters:')
print(f'Kappa: {kappa_opt}')
print(f'Theta: {theta_opt}')
print(f'Sigma: {sigma_opt}')

# Simulate O-U process with optimized parameters
simulated_ou_process = ornstein_uhlenbeck([kappa_opt, theta_opt, sigma_opt/10], spread)

# Generate buy/sell signals for each stock based on the O-U process simulation
buy_signal_x1 = np.where(x1 < simulated_ou_process, 1, 0)
sell_signal_x1 = np.where(x1 > simulated_ou_process, -1, 0)

buy_signal_x2 = np.where(x2 < simulated_ou_process, 1, 0)
sell_signal_x2 = np.where(x2 > simulated_ou_process, -1, 0)

# Find the index where correlation turns negative
rolling_corr = x1.rolling(window=40).corr(x2)

# Find the index where the correlation is below 0
correlation_breakpoints = rolling_corr[rolling_corr < 0].index

# Plot the actual stock prices, O-U process simulations, and buy/sell signals
plt.figure(figsize=(12, 8))

# Plot Stock 1
plt.subplot(2, 1, 1)
plt.plot(x1, label='Actual Stock 1 Price', linewidth=2)
plt.plot(simulated_ou_process, label='Simulated O-U Process for Stock 1', linestyle='--', linewidth=2)
plt.scatter(np.where(buy_signal_x1 == 1)[0], x1[buy_signal_x1 == 1], marker='^', color='g', label='Buy Signal')
plt.scatter(np.where(sell_signal_x1 == -1)[0], x1[sell_signal_x1 == -1], marker='v', color='r', label='Sell Signal')
plt.axvline(x=correlation_breakpoints[0], color='yellow', linestyle='--', label='Correlation Turns Negative')
plt.xlabel('Time')
plt.ylabel('Stock 1 Price')
plt.title('Stock 1 Price and Simulated O-U Process with Buy/Sell Signals')
plt.legend()
plt.xlim(correlation_breakpoints[0] - 100, correlation_breakpoints[0] + 100)

# Plot Stock 2
plt.subplot(2, 1, 2)
plt.plot(x2, label='Actual Stock 2 Price', linewidth=2)
plt.plot(simulated_ou_process, label='Simulated O-U Process for Stock 2', linestyle='--', linewidth=2)
plt.scatter(np.where(buy_signal_x2 == 1)[0], x2[buy_signal_x2 == 1], marker='^', color='g', label='Buy Signal')
plt.scatter(np.where(sell_signal_x2 == -1)[0], x2[sell_signal_x2 == -1], marker='v', color='r', label='Sell Signal')
plt.axvline(x=correlation_breakpoints[0], color='yellow', linestyle='--', label='Correlation Turns Negative')
plt.xlabel('Time')
plt.ylabel('Stock 2 Price')
plt.title('Stock 2 Price and Simulated O-U Process with Buy/Sell Signals')
plt.legend()
plt.xlim(correlation_breakpoints[0] - 100, correlation_breakpoints[0] + 100)

plt.tight_layout()
plt.show()


#%%


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Assuming you have x1 and x2 defined
x1 = Ari_sample.copy()
x2 = Fbrt_sample.copy()

# Date range from June 1, 2019, to June 1, 2024
date_range = pd.date_range(start='2019-06-01', end='2024-06-01', freq='D')


k0 = np.mean(x1/x2)
# Calculate the ratio
ratio = x1 / x2

# Calculate the 40-day moving average
moving_average = ratio.rolling(window=40).mean()

k0 = moving_average.fillna(np.mean(moving_average.fillna(0)))
# Calculate the spread
spread = x1/(k0*x2)


# Define the O-U process
def ornstein_uhlenbeck(theta, x):
    kappa, theta_ou, sigma = theta
    dt = 1  # Assuming daily data

    ou_process = np.zeros_like(x)
    ou_process[0] = x.iloc[0]

    for t in range(1, len(x)):
        dW = np.random.normal(0, np.sqrt(dt))
        ou_process[t] = (
            ou_process[t - 1] * np.exp(-kappa * dt)
            + theta_ou * (1 - np.exp(-kappa * dt))
            + sigma * np.sqrt((1 - np.exp(-2 * kappa * dt)) / (2 * kappa)) * dW
        )

    return ou_process

# Define the log-likelihood function for O-U process
def log_likelihood_OU(theta, x):
    ou_process = ornstein_uhlenbeck(theta, x)
    residuals = x - ou_process
    sigma2 = np.var(residuals)
    log_likelihood_val = -len(x) * np.log(np.sqrt(2 * np.pi * sigma2)) - np.sum(residuals ** 2) / (2 * sigma2)
    return -log_likelihood_val

# Set up optimization
initial_guess = [1, np.mean(spread), np.std(spread)]
bounds = [(0, None), (None, None), (0, None)]  # Kappa and Sigma must be positive

# Perform optimization
result = minimize(log_likelihood_OU, initial_guess, args=(spread,), bounds=bounds)

# Extract optimized parameters
kappa_opt, theta_opt, sigma_opt = result.x

# Display the optimized parameters
print('Optimized Parameters:')
print(f'Kappa: {kappa_opt}')
print(f'Theta: {theta_opt}')
print(f'Sigma: {sigma_opt}')

# Simulate O-U process with optimized parameters
simulated_ou_process = ornstein_uhlenbeck([kappa_opt, theta_opt, sigma_opt], spread)

# Generate buy/sell signals for each stock based on the O-U process simulation
buy_signal_x1 = np.where(x1 < simulated_ou_process, 1, 0)
sell_signal_x1 = np.where(x1 > simulated_ou_process, -1, 0)

buy_signal_x2 = np.where(x2 < simulated_ou_process, 1, 0)
sell_signal_x2 = np.where(x2 > simulated_ou_process, -1, 0)

# Find the index where correlation turns negative
rolling_corr = x1.rolling(window=40).corr(x2)

# Find the index where the correlation is below 0
correlation_breakpoints = rolling_corr[rolling_corr < 0].index

# Define a function to calculate profit from pair trading
def calculate_profit(initial_investment, buy_signal, sell_signal, stock_price):
    position = 0  # 0: no position, 1: long position, -1: short position
    cash = initial_investment
    stocks = 0
    days_to_hold = 40

    for i in range(len(buy_signal)):
        if buy_signal[i] == 1 and position == 0:
            position = 1
            stocks = cash / stock_price[i]
            cash = 0
        elif sell_signal[i] == -1 and position == 0:
            position = -1
            stocks = -cash / stock_price[i]
            cash = 0
        elif (sell_signal[i] == -1 and position == 1) or (buy_signal[i] == 1 and position == -1):
            position = 0
            cash = stocks * stock_price[i]
            stocks = 0
        elif i - days_to_hold >= 0 and position != 0:
            position = 0
            cash = stocks * stock_price[i]
            stocks = 0

    return cash + stocks * stock_price[-1]

# Calculate profit for Stock 1
profit_x1 = calculate_profit(100000, buy_signal_x1.loc[date_range], sell_signal_x1.loc[date_range], x1.loc[date_range])

# Calculate profit for Stock 2
profit_x2 = calculate_profit(100000, buy_signal_x2.loc[date_range], sell_signal_x2.loc[date_range], x2.loc[date_range])

# Print profits
print(f'Profit from pair trading ARI: ${profit_x1:.2f}')
print(f'Profit from pair trading BMXT 2: ${profit_x2:.2f}')
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Assuming you have x1 and x2 defined
x1 = Ari_sample.copy()
x2 = Fbrt_sample.copy()

# Date range from June 1, 2019, to June 1, 2024
date_range = pd.date_range(start='2019-06-01', periods=len(x1), freq='D')

k0 = np.mean(x1/x2)
# Calculate the spread
spread = x1/(k0*x2)
spread = x1 - x2

# Calculate the ratio
ratio = x1 / x2

# Calculate the 40-day moving average
moving_average = ratio.rolling(window=40).mean()

k0 = moving_average.fillna(np.mean(moving_average.fillna(0)))
#%%
spread = x1/(k0*x2)
# Define the O-U process
def ornstein_uhlenbeck(theta, x):
    kappa, theta_ou, sigma = theta
    dt = 1  # Assuming daily data

    ou_process = np.zeros_like(x)
    ou_process[0] = x.iloc[0]

    for t in range(1, len(x)):
        dW = np.random.normal(0, np.sqrt(dt))
        ou_process[t] = (
            ou_process[t - 1] * np.exp(-kappa * dt)
            + theta_ou * (1 - np.exp(-kappa * dt))
            + sigma * np.sqrt((1 - np.exp(-2 * kappa * dt)) / (2 * kappa)) * dW
        )

    return ou_process

# Define the log-likelihood function for O-U process
def log_likelihood_OU(theta, x):
    ou_process = ornstein_uhlenbeck(theta, x)
    residuals = x - ou_process
    sigma2 = np.var(residuals)
    log_likelihood_val = -len(x) * np.log(np.sqrt(2 * np.pi * sigma2)) - np.sum(residuals ** 2) / (2 * sigma2)
    return -log_likelihood_val

# Set up optimization
initial_guess = [1, np.mean(spread), np.std(spread)]
bounds = [(0, None), (None, None), (0, None)]  # Kappa and Sigma must be positive

# Perform optimization
result = minimize(log_likelihood_OU, initial_guess, args=(spread,), bounds=bounds)

# Extract optimized parameters
kappa_opt, theta_opt, sigma_opt = result.x

# Display the optimized parameters
print('Optimized Parameters:')
print(f'Kappa: {kappa_opt}')
print(f'Theta: {theta_opt}')
print(f'Sigma: {sigma_opt}')

# Simulate O-U process with optimized parameters
simulated_ou_process = ornstein_uhlenbeck([kappa_opt, theta_opt, sigma_opt], spread)

# Generate buy/sell signals for each stock based on the O-U process simulation
buy_signal_x1 = np.where(x1 < simulated_ou_process, 1, 0)
sell_signal_x1 = np.where(x1 > simulated_ou_process, -1, 0)

buy_signal_x2 = np.where(x2 < simulated_ou_process, 1, 0)
sell_signal_x2 = np.where(x2 > simulated_ou_process, -1, 0)

# Convert arrays to Pandas Series with the same length as date_range
buy_signal_x1 = pd.Series(buy_signal_x1, index=date_range)
sell_signal_x1 = pd.Series(sell_signal_x1, index=date_range)
x1_series = pd.Series(x1.values, index=date_range)

buy_signal_x2 = pd.Series(buy_signal_x2, index=date_range)
sell_signal_x2 = pd.Series(sell_signal_x2, index=date_range)
x2_series = pd.Series(x2.values, index=date_range)

# Find the index where correlation turns negative
rolling_corr = x1_series.rolling(window=40).corr(x2_series)

# Find the index where the correlation is below 0
correlation_breakpoints = rolling_corr[rolling_corr < 0].index

# Define a function to calculate profit from pair trading
def calculate_profit(initial_investment, buy_signal, sell_signal, stock_price):
    position = 0  # 0: no position, 1: long position, -1: short position
    cash = initial_investment
    stocks = 0
    days_to_hold = 40

    for i in range(len(buy_signal)):
        if buy_signal[i] == 1 and position == 0:
            position = 1
            stocks = cash / stock_price[i]
            cash = 0
        elif sell_signal[i] == -1 and position == 0:
            position = -1
            stocks = -cash / stock_price[i]
            cash = 0
        elif (sell_signal[i] == -1 and position == 1) or (buy_signal[i] == 1 and position == -1):
            position = 0
            cash = stocks * stock_price[i]
            stocks = 0
        elif i - days_to_hold >= 0 and position != 0:
            position = 0
            cash = stocks * stock_price[i]
            stocks = 0

    return cash + stocks * stock_price[-1]

# Calculate profit for Stock 1
profit_x1 = calculate_profit(100000, buy_signal_x1, sell_signal_x1, x1_series)

# Calculate profit for Stock 2
profit_x2 = calculate_profit(100000, buy_signal_x2, sell_signal_x2, x2_series)

# Print profits
print(f'Profit from pair trading Stock 1: ${profit_x1:.2f}')
print(f'Profit from pair trading Stock 2: ${profit_x2:.2f}')
#%%


# Plot the actual spread, the simulated O-U process, buy/sell signals, correlation breakpoints, mean spread, and stock prices
plt.figure(figsize=(10, 6))
plt.plot(spread, label='Actual Spread', linewidth=2)
plt.plot(simulated_ou_process, label='Simulated O-U Process', linestyle='--', linewidth=2)
plt.scatter(buy_signal_x1.index[buy_signal_x1 == 1], spread[buy_signal_x1 == 1], marker='^', color='g', label='Buy Signal')
plt.scatter(sell_signal_x1.index[sell_signal_x1 == -1], spread[sell_signal_x1 == -1], marker='v', color='r', label='Sell Signal')
if not correlation_breakpoints.empty:
    plt.scatter(correlation_breakpoints, spread.loc[correlation_breakpoints], marker='o', color='y', label='Correlation Breakpoint')
    plt.axvline(x=correlation_breakpoints[0], color='yellow', linestyle='--', label='Correlation Turns Negative')
plt.axhline(y=np.mean(spread), color='orange', linestyle='-', label='Mean Spread')

# Adjust the x-axis limits to zoom in on the correlation break
if not correlation_breakpoints.empty:
    plt.xlim(correlation_breakpoints[0] - 500, correlation_breakpoints[0] + 500)

plt.xlabel('Date')
plt.ylabel('Spread')
plt.title('Actual Spread vs. Simulated O-U Process with Buy/Sell Signals and Correlation Breakpoints')
plt.legend()
plt.xticks(rotation=45, ha='right')  # Rotate and align the x-axis labels
plt.tight_layout()  # Ensure tight layout for better visibility
plt.show()

#%%
