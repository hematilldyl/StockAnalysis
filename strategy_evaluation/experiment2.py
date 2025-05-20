from StrategyLearner import StrategyLearner
from ManualStrategy import ManualStrategy
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from marketsimcode import compute_portvals
from util import get_data, plot_data
import numpy as np

def author():
    return 'dhematillake3'
def run_exp2():
    sl = StrategyLearner()
    sl.add_evidence(symbol="JPM",
                sd=dt.datetime(2008, 1, 1),
                ed=dt.datetime(2009, 12, 31),
                sv=100000)

    os_learner_orders = sl.testPolicy(symbol="JPM",
                sd=dt.datetime(2008, 1, 1),
                ed=dt.datetime(2009, 12, 31),
                sv=100000)


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

    impacts = [0,0.005,0.01,0.05,0.1]
    sharpe= []
    cum_ret = []
    for imp in impacts:
        benchmark_portfolio = compute_portvals(data,symbol='JPM',start_val=100000,commission=0,impact=imp)
        os_learner_trades = compute_portvals(os_learner_orders,symbol='JPM',start_val=100000,commission=0,impact=imp)
        values_learner = np.sum(os_learner_trades, axis=1)
        daily_rets = (values_learner .iloc[1:] / values_learner .iloc[:-1].values) - 1
        avg_daily_rets = daily_rets.mean()
        std_daily_rets = daily_rets.std()
        cum_ret.append((values_learner .iloc[-1] / values_learner .iloc[0]) - 1)
        sharpe.append(np.sqrt(252) * avg_daily_rets / std_daily_rets)
        plt.plot(values_learner / values_learner.iloc[0], label=str(imp))
    np.savetxt('exp2stats.txt',[impacts,cum_ret,sharpe])
    plt.legend()
    plt.xticks(rotation=20)
    plt.title('In Sample Impact Comparison with 0 Commission')
    plt.ylabel('Normalized Value (Dollar/Dollar at Start)')
    plt.savefig('images/experiment2.png')