
from marketsimcode import compute_portvals
from TheoreticallyOptimalStrategy import testPolicy
import numpy as np
import matplotlib.pyplot as plt
from indicators import crosses, stochasticoscillator,momentum,bollingerbandpercent,percentagepriceindicator
from util import get_data, plot_data
import pandas as pd
import datetime as dt

def author():
    return 'dhematillake3'

if __name__ == '__main__':

    #TOS Policy Comparison
    orders = testPolicy()
    orders['Symbol'] = 'JPM'
    orders = orders.dropna()
    portfolio = compute_portvals(orders,start_val=100000,commission=0,impact=0)
    benchmark = orders.copy()
    benchmark['Order'] = np.nan
    benchmark['Shares'].iloc[0] = 1000
    benchmark['Order'].iloc[0] = 'BUY'
    benchmark['Shares'].iloc[-1] = 1000
    benchmark['Order'].iloc[-1] = 'SELL'
    benchmark = benchmark.dropna()
    benchmark_portfolio = compute_portvals(benchmark,start_val=100000,commission=0,impact=0)
    benchmark_value = benchmark_portfolio.sum(axis=1)
    TOS_value = portfolio.sum(axis=1)

    bm_cumret = (benchmark_value.iloc[-1]/benchmark_value.iloc[0])-1
    TOS_cumret = (TOS_value.iloc[-1] / TOS_value.iloc[0])-1

    bm_dailyret = (benchmark_value.iloc[1:]/benchmark_value.iloc[:-1].values)-1
    bm_stdret = bm_dailyret.std()
    bm_avgret = bm_dailyret.mean()
    bm_sharpe = np.sqrt(252)*bm_avgret/bm_stdret
    TOS_dailyret = (TOS_value.iloc[1:] / TOS_value.iloc[:-1].values) - 1
    TOS_stdret = TOS_dailyret.std()
    TOS_avgret = TOS_dailyret.mean()
    TOS_sharpe = np.sqrt(252) * TOS_avgret / TOS_stdret

    benchmark_value = benchmark_value/benchmark_value.iloc[0]

    TOS_value = TOS_value/TOS_value.iloc[0]
    benchmark_value.plot(label='Benchmark',color='tab:purple')
    TOS_value.plot(label='Portfolio',color="tab:red")
    plt.ylabel('Normalized Value (-)')
    plt.xlabel('Date')
    plt.legend()
    plt.savefig('TOS.png')
    statistics = np.zeros((4,2))
    statistics[0,0] = bm_cumret
    statistics[0,1] = TOS_cumret

    statistics[1,0] = bm_avgret
    statistics[1,1] = TOS_avgret

    statistics[2,0] = bm_stdret
    statistics[2,1] = TOS_stdret

    statistics[3,0] = bm_sharpe
    statistics[3,1] = TOS_sharpe

    np.savetxt('summary_stats.txt',statistics)

    
    #start technical indicators
    symbol = 'JPM'
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    data = get_data([symbol], pd.date_range(sd, ed))
    cross = pd.Series(crosses(data['JPM']))
    death_cross = cross[cross>0]
    golden_cross = cross[cross<0]
    SO = stochasticoscillator(data['JPM'])
    M = momentum(data['JPM'])
    BBP = bollingerbandpercent(data['JPM'])
    PPO = percentagepriceindicator(data['JPM'])
    indicators = pd.concat([data,cross,SO,M,BBP,PPO],axis=1)
    indicators.columns = ['SPY','JPM','Crosses','SO','M','BBP','PPO']

    fig,axes = plt.subplots(2)
    indicators[['JPM','BBP']].plot(subplots=True,ax=axes)
    #Plotting BB for visualization purposes only
    BBU = indicators['JPM'].rolling(20).mean()+2*indicators['JPM'].rolling(20).std()
    BBL = indicators['JPM'].rolling(20).mean() - 2 * indicators['JPM'].rolling(20).std()
    BBM = indicators['JPM'].rolling(20).mean()
    axes[0].plot(BBU,label='Upper Band',color='r')
    axes[0].plot(BBL,label='Lower Band',color='r')
    axes[0].plot(BBM, label='SMA 20', color='g')
    axes[0].legend()
    axes[1].axhline(y=0.8)
    axes[1].axhline(y=0.2)
    axes[0].set_title('Bollinger Band Percent Indicator')
    axes[0].set_ylabel('Price ($)')
    axes[1].set_ylabel('Bollinger Band Percent (%)')
    axes[1].set_xlabel('Date')
    plt.savefig('BBP.png')

    fig,axes = plt.subplots(2)
    indicators[['JPM','Crosses']].plot(subplots=True,ax=axes)
    #Calling a SMA for illustrative purposes only
    SMA25 = indicators['JPM'].rolling(25).mean()
    SMA100 = indicators['JPM'].rolling(100).mean()
    axes[0].plot(SMA25,label='SMA 25',color='r')
    axes[0].plot(SMA100,label='SMA 100',color='g')
    axes[0].set_title('Golden and Death Cross Indicator (SMA = 25, 100 periods)')
    axes[0].set_ylabel('Price ($)')
    axes[1].set_ylabel('25 Day - 100 Day SMA Cross Flag')
    axes[1].set_xlabel('Date')
    axes[0].legend()
    plt.savefig('crosses.png')

    fig,axes = plt.subplots(2)
    indicators[['JPM','SO']].plot(subplots=True,ax=axes)
    axes[0].set_title('Fast Stochastic Indicator (14 Day Window)')
    axes[0].legend()
    axes[1].axhline(y=80)
    axes[1].axhline(y=20)
    axes[0].set_ylabel('Price ($)')
    axes[1].set_ylabel('Stochastic Indicator (-)')
    axes[1].set_xlabel('Date')
    plt.savefig('stochastic.png')

    fig,axes = plt.subplots(2)
    indicators[['JPM','M']].plot(subplots=True,ax=axes)
    axes[0].set_title('Momentum Indicator (14 Day Window)')
    axes[0].legend()
    axes[1].axhline(y=0)
    axes[0].set_ylabel('Price ($)')
    axes[1].set_ylabel('Momentum (-)')
    axes[1].set_xlabel('Date')
    plt.savefig('momentum.png')

    fig,axes = plt.subplots(2,figsize=(8,5))
    #Calculate PPO and Signal lines for illustrative purposes
    EMA12 = data['JPM'].ewm(span=12,adjust=False).mean()
    EMA26 = data['JPM'].ewm(span=26,adjust=False).mean()
    PPO = (EMA12-EMA26)/EMA26*100
    SIGNAL = PPO.ewm(span=9,adjust=False).mean()

    axes[0].set_title('Percentage Price Indicator (EMA = 12, 26 periods)')
    axes[0].plot(indicators['JPM'])
    axes[0].legend()
    axes[1].bar(indicators['PPO'].index,indicators['PPO'].values,label='PPI-SIGNAL')
    axes[1].plot(PPO,label='PPI')
    axes[1].plot(SIGNAL, label='Signal Line')
    axes[1].legend()
    axes[1].axhline(y=0)
    axes[0].set_ylabel('Price ($)')
    axes[1].set_ylabel('Percentage Price Indicator (-)')
    axes[1].set_xlabel('Date')
    plt.savefig('PPO.png')





