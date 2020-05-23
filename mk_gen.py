import numpy as np
import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt

typeIn = input("Enter a portfolio, separate by ',': ")

while (typeIn[-1].isalpha() == False):
    typeIn = input("Enter again: ")
assets = typeIn.upper().split(',')

yr = input("Please enter starting year: ")
yr = ''.join([yr, '-1-1'])

rf = float(input("Enter risk free asset return: "))

pf_data = pd.DataFrame()

for a in assets:
    pf_data[a] = data.DataReader(a, data_source='yahoo', 
                                 start=yr)['Adj Close']

log_returns = np.log(1 + pf_data.pct_change())

pfolio_returns = []
pfolio_volatilities = []
p_weights = []

for x in range(10000):
    weights = np.random.random(len(assets))
    weights /= sum(weights)
    p_weights.append(weights)
    pfolio_returns.append(np.dot(weights,
                                 log_returns.mean() * 250))
    pfolio_volatilities.append(np.dot(weights.T, 
                                     np.dot(log_returns.cov() * 250, 
                                            weights)) ** 0.5)
pfolio_returns = np.array(pfolio_returns)
pfolio_volatilities = np.array(pfolio_volatilities)

dta = {'Return': pfolio_returns,'Volatility':pfolio_volatilities}

for counter, symbol in enumerate(pf_data.columns.tolist()):
    dta[symbol+ "_weight"] = [w[counter] for w in p_weights]
    
portfolios = pd.DataFrame(dta)
sharpe_ratio = (portfolios['Return'] - rf) / portfolios['Volatility']

op_portfolio = portfolios.iloc[(sharpe_ratio).idxmax()]
min_risk = portfolios.iloc[portfolios['Volatility'].idxmin()]
max_return = portfolios.iloc[portfolios['Return'].idxmax()]

print("Optimal Portfolio:")
print(op_portfolio)
print("Low Risk Portfolio:")
print(min_risk)
print("Maximum Return Portfolio:")
print(max_return)

choice = input("Do you want to see the graph? (y/n): ")

if (choice.upper() == 'Y'):
    portfolios.plot(x='Volatility', y='Return',
                kind='scatter', figsize=(20,10))
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.show()
