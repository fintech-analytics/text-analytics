import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

df = pd.read_csv('HSI.csv')
df.fillna(method='ffill',inplace=True)
df[df['Adj Close'].isnull() == True]

# Compute the log return
def log_return(x,x_lag_1):
    logreturn = np.log(x) - np.log(x_lag_1)
    return logreturn

df['log_return'] = np.nan

for i in range(1, len(df)):
    x = df.loc[i,'Adj Close']
    x_1 = df.loc[i-1, 'Adj Close']
    df.loc[i,'log_return'] = log_return(x,x_1)*100

df.set_index('Date',inplace=True)
df.index = pd.to_datetime(df.index)
x = df[['log_return']].dropna()

# Fit into the arch model
m1 = arch_model(x).fit()
std_resid = m1.std_resid.to_frame()
volatility = m1.conditional_volatility

# Format the volatility measures we need
df['std_resid'] = std_resid
df['abs_resid'] = np.abs(std_resid)
df['sq_resid'] = std_resid**2
df['volatility'] = volatility
df.to_csv('HSI_with_GARCH_resid.csv')