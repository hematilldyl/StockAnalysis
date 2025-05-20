import pandas as pd
import numpy as np
from util import get_data, plot_data

def author():
    return 'dhematillake3'
def crosses(prices):
    SMA25 = prices.rolling(window=25).mean()
    SMA100 = prices.rolling(window=100).mean()
    MA = np.zeros((SMA25.shape[0],2))
    MA[:,0] = SMA25.values
    MA[:,1] = SMA100.values
    MAs = pd.DataFrame(index = SMA25.index,data=MA,columns=['SMA25','SMA100'])
    # death cross line of code derived from https://thepythonyouneed.com/how-to-compute-the-death-cross-with-pandas-in-python/
    MAs["death_cross"] = MAs.apply(lambda row: 1 if row[f"SMA25"] < row[f"SMA100"]  else 0, axis=1)
    MAs["crosses"] = MAs['death_cross'].diff()
    cross = MAs['crosses']

    return MAs['crosses']

def stochasticoscillator(prices,lookback=14):
    L14 = prices.rolling(lookback).min()
    H14 = prices.rolling(lookback).max()
    K = (prices.iloc[14:]-L14)/(H14-L14)*100
    return K

def momentum(prices):
    momentum = prices/prices.shift(14)-1
    return momentum

def bollingerbandpercent(prices):
    BBU = prices.rolling(20).mean() +2*prices.rolling(20).std()
    BBL = prices.rolling(20).mean()-2*prices.rolling(20).std()
    BBP = (prices - BBL)/(BBU-BBL)
    return BBP

def percentagepriceindicator(prices):
    EMA12 = prices.ewm(span=12,adjust=False).mean()
    EMA26 = prices.ewm(span=26,adjust=False).mean()
    PPO = (EMA12-EMA26)/EMA26*100
    SIGNAL = PPO.ewm(span=9,adjust=False).mean()
    return PPO-SIGNAL