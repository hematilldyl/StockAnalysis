from StrategyLearner import StrategyLearner
from ManualStrategy import ManualStrategy
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from marketsimcode import compute_portvals
from util import get_data, plot_data
import numpy as np

def run_exp1():
    np.random.seed(903741146)
    sd=dt.datetime(2008, 1, 1)
    ed=dt.datetime(2009, 12, 31)

    data = get_data(['JPM'], pd.date_range(sd, ed))
    data['Order'] = np.nan
    data['Shares'] = np.nan
    data['Order'].iloc[0]='BUY'
    data['Shares'].iloc[0]=1000
    data['Symbol']='JPM'
    data['Order'].iloc[-1] = 'HOLD'
    data['Shares'].iloc[-1] = 1000
    data = data.dropna()
    benchmark_portfolio = compute_portvals(data,symbol='JPM',start_val=100000,commission=9.95,impact=0.005)

    ms = ManualStrategy(commission=9.95,impact=0.005)
    is_manual_trades = ms.train_policy()
    os_manual_trades = ms.test_policy()

    sl = StrategyLearner()
    sl.add_evidence(symbol="JPM",
                sd=dt.datetime(2008, 1, 1),
                ed=dt.datetime(2009, 12, 31),
                sv=100000)
    is_learner_orders = sl.testPolicy(symbol="JPM",
                sd=dt.datetime(2008, 1, 1),
                ed=dt.datetime(2009, 12, 31),
                sv=100000)

    is_learner_trades = compute_portvals(is_learner_orders,symbol='JPM',start_val=100000,commission=9.95,impact=0.005)
    os_learner_orders = sl.testPolicy()
    os_learner_trades = compute_portvals(os_learner_orders,symbol='JPM',start_val=100000,commission=9.95,impact=0.005)

    values = np.sum(is_manual_trades, axis=1)
    values_learner = np.sum(is_learner_trades, axis=1)
    benchmark = np.sum(benchmark_portfolio, axis=1)
    plt.figure()
    plt.plot(values / values.iloc[0], label='Manual Strategy', color='red')
    plt.plot(values_learner / values_learner.iloc[0], label='Learner Strategy', color='blue')
    plt.plot(benchmark / benchmark.iloc[0], label='Benchmark', color='purple')
    plt.legend()
    plt.xticks(rotation=20)
    plt.title('In Sample Strategy Comparison')
    plt.ylabel('Normalized Value (Dollar/Dollar at Start)')
    plt.savefig('images/insamplecomparisonexperiment1.png')

    sd=dt.datetime(2010, 1, 1)
    ed=dt.datetime(2011, 12, 31)
    data = get_data(['JPM'], pd.date_range(sd, ed))
    data['Order'] = np.nan
    data['Shares'] = np.nan
    data['Order'].iloc[0]='BUY'
    data['Shares'].iloc[0]=1000
    data['Symbol']='JPM'
    data['Order'].iloc[-1] = 'HOLD'
    data['Shares'].iloc[-1] = 1000
    data = data.dropna()
    benchmark_portfolio = compute_portvals(data,symbol='JPM',start_val=100000,commission=9.95,impact=0.005)

    values = np.sum(os_manual_trades, axis=1)
    values_learner = np.sum(os_learner_trades, axis=1)
    benchmark = np.sum(benchmark_portfolio, axis=1)
    plt.figure()
    plt.plot(values / values.iloc[0], label='Manual Strategy', color='red')
    plt.plot(values_learner / values_learner.iloc[0], label='Learner Strategy', color='blue')
    plt.plot(benchmark / benchmark.iloc[0], label='Benchmark', color='purple')
    plt.legend()
    plt.xticks(rotation=20)
    plt.title('Out of Sample Strategy Comparison')
    plt.ylabel('Normalized Value (Dollar/Dollar at Start)')
    plt.savefig('images/outofsamplecomparisonexperiment1.png')
    return
