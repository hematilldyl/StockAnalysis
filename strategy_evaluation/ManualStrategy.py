import datetime as dt
import random
import numpy as np
import pandas as pd
from util import get_data, plot_data
from indicators import crosses, stochasticoscillator,momentum,bollingerbandpercent,percentagepriceindicator
import matplotlib.pyplot as plt
from marketsimcode import compute_portvals

class ManualStrategy(object):
    """
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output.
    :type verbose: bool
    :param impact: The market impact of each transaction, defaults to 0.0
    :type impact: float
    :param commission: The commission amount charged, defaults to 0.0
    :type commission: float
    """
    def author(self):
        return 'dh'
    # constructor
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """
        Constructor method
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission


    def train_policy(
            self,
            symbol="JPM",
            sd=dt.datetime(2008, 1, 1),
            ed=dt.datetime(2009, 12, 31),
            sv=100000,
    ):
        """
        Tests your manual over a given time frame.

        :param symbol: The stock symbol to train on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        """
        data = get_data([symbol], pd.date_range(sd, ed))
        data['Order'] = np.nan
        data['Shares'] = np.nan
        data['Crosses'] = crosses(data[symbol]).values #+1 means exit -1 means enter
        data['BBP'] = bollingerbandpercent(data[symbol]).values
        data['Momentum'] = momentum(data[symbol]).values
        data['SO'] = stochasticoscillator(data[symbol]).values
        data['PPO'] = percentagepriceindicator(data[symbol]).values

        data['Order'].iloc[0] = 'HOLD'
        data['Shares'].iloc[0] = 0
        data['Order'].iloc[-1] = 'HOLD'
        data['Shares'].iloc[-1] = 0
        mask = (data['Crosses'] == 1) | (data['BBP'] > 0.8) & (data['Momentum'] > 0.2) & (data['SO'] > 80) & (data['PPO']>2)
        data['Order'][mask] = 'SELL'
        mask = (data['Crosses'] == -1) | (data['BBP'] < 0.2) & (data['Momentum'] < -0.2) & (data['SO'] < 20) & (data['PPO']<-2)
        data['Order'][mask] = 'BUY'
        orders = data[['Order','Shares']]
        orders.loc[orders['Order']=='SELL',"Shares"]=1000
        orders.loc[orders['Order']=='BUY',"Shares"]=1000

        orders = orders.reset_index()
        orders=orders.dropna()
        f = orders['Order'].ne(orders['Order'].shift()).cumsum()
        orders=orders.groupby(f).first()
        orders = orders.set_index('index')
        orders['Shares'].iloc[2:]=orders['Shares'].iloc[2:]*2
        orders['Symbol']=symbol
        trades = compute_portvals(orders,symbol=symbol,start_val=100000,commission=self.commission,impact=self.impact)

        bm = get_data(['JPM'], pd.date_range(sd, ed))
        bm['Order'] = np.nan
        bm['Shares'] = np.nan
        bm['Order'].iloc[0] = 'BUY'
        bm['Shares'].iloc[0] = 1000
        bm['Symbol'] = 'JPM'
        bm['Order'].iloc[-1] = 'HOLD'
        bm['Shares'].iloc[-1] = 1000
        bm= bm.dropna()
        benchmark_portfolio = compute_portvals(bm, symbol='JPM', start_val=100000, commission=self.commission, impact=self.impact)

        values = np.sum(trades, axis=1)
        benchmark = np.sum(benchmark_portfolio, axis=1)
        plt.plot(values / values.iloc[0], label='Manual Strategy',color='red')
        plt.plot(benchmark / benchmark.iloc[0], label='Benchmark',color='purple')
        for order in orders[orders['Order']=='BUY'].index:
            plt.axvline(x=order,color='blue')
        for order in orders[orders['Order']=='SELL'].index:
            plt.axvline(x=order,color='black')
        plt.legend()
        plt.xticks(rotation=20)
        plt.title('In Sample Manual Strategy')
        plt.ylabel('Normalized Value (Dollar/Dollar at Start)')
        plt.savefig('images/InSampleManual.png')
        return trades

    def test_policy(
            self,
            symbol="JPM",
            sd=dt.datetime(2010, 1, 1),
            ed=dt.datetime(2011, 12, 31),
            sv=100000,
    ):
        """
        Tests your manual over a given time frame.

        :param symbol: The stock symbol to train on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        """
        data = get_data([symbol], pd.date_range(sd, ed))
        data['Order'] = np.nan
        data['Shares'] = np.nan
        data['Crosses'] = crosses(data[symbol]).values  # +1 means exit -1 means enter
        data['BBP'] = bollingerbandpercent(data[symbol]).values
        data['Momentum'] = momentum(data[symbol]).values
        data['PPO'] = percentagepriceindicator(data[symbol]).values
        data['SO'] = stochasticoscillator(data[symbol]).values
        data['Order'].iloc[0] = 'HOLD'
        data['Shares'].iloc[0] = 0
        data['Order'].iloc[-1] = 'HOLD'
        data['Shares'].iloc[-1] = 0
        mask = (data['Crosses'] == 1) | (data['BBP'] > 0.8) & (data['Momentum'] > 0.2) & (data['SO'] > 80) & (
                    data['PPO'] > 2)
        data['Order'][mask] = 'SELL'
        mask = (data['Crosses'] == -1) | (data['BBP'] < 0.2) & (data['Momentum'] < -0.2) & (data['SO'] < 20) & (
                    data['PPO'] < -2)
        data['Order'][mask] = 'BUY'
        orders = data[['Order', 'Shares']]
        orders.loc[orders['Order'] == 'SELL', "Shares"] = 1000
        orders.loc[orders['Order'] == 'BUY', "Shares"] = 1000

        orders = orders.reset_index()
        orders = orders.dropna()
        f = orders['Order'].ne(orders['Order'].shift()).cumsum()
        orders = orders.groupby(f).first()
        orders = orders.set_index('index')
        orders['Shares'].iloc[2:] = orders['Shares'].iloc[2:]*2
        orders['Symbol'] = symbol
        trades = compute_portvals(orders, symbol=symbol, start_val=100000, commission=self.commission, impact=self.impact)

        bm = get_data(['JPM'], pd.date_range(sd, ed))
        bm['Order'] = np.nan
        bm['Shares'] = np.nan
        bm['Order'].iloc[0] = 'BUY'
        bm['Shares'].iloc[0] = 1000
        bm['Symbol'] = 'JPM'
        bm['Order'].iloc[-1] = 'HOLD'
        bm['Shares'].iloc[-1] = 1000
        bm = bm.dropna()
        benchmark_portfolio = compute_portvals(bm, symbol='JPM', start_val=100000, commission=self.commission, impact=self.impact)

        values = np.sum(trades, axis=1)
        benchmark = np.sum(benchmark_portfolio, axis=1)
        plt.figure()
        plt.plot(values / values.iloc[0], label='Manual Strategy', color='red')
        plt.plot(benchmark / benchmark.iloc[0], label='Benchmark', color='purple')
        for order in orders[orders['Order'] == 'BUY'].index:
            plt.axvline(x=order, color='blue')
        for order in orders[orders['Order'] == 'SELL'].index:
            plt.axvline(x=order, color='black')
        plt.legend()
        plt.xticks(rotation=20)
        plt.title('Out of Sample Manual Strategy')
        plt.ylabel('Normalized Value (Dollar/Dollar at Start)')
        plt.savefig('images/OutSampleManual.png')

        return trades

