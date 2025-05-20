import datetime as dt

import numpy as np

import pandas as pd
from util import get_data, plot_data

def author():
    return 'dhematillake3'

def compute_portvals(orders=None,
        symbol='JPM',
        start_val=1000000,
        commission=9.95,
        impact=0.005,
):
    """
    Computes the portfolio values.
  	Follows steps given by Prof. Tucker Balch.

    :param orders_file: Path of the order file or the file object
    :type orders_file: str or file object
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input

    orders = orders.sort_index()
    start_date = orders.index[0]
    end_date = orders.index[-1]
    reference = get_data(['SPY'], pd.date_range(start_date, end_date))
    prices = np.zeros((reference.shape[0], len(orders['Symbol'].unique()) + 1))
    prices[:, -1] = 1
    inc = 0
    ticker_data = get_data([symbol], pd.date_range(start_date, end_date))
    prices[:,0] = ticker_data[symbol]
    prices = pd.DataFrame(index=reference.index, data=prices, columns=[symbol,'Cash'])
    trades = prices.copy()
    for col in trades.columns:
        trades[col] = np.nan

    cash = start_val
    for i in range(0, len(orders)):
        date = orders.index[i]
        ticker = orders['Symbol'].iloc[i]
        order_type = orders['Order'].iloc[i]
        order_number = orders['Shares'].iloc[i]
        buy = 1
        if order_type == 'SELL':
            buy = -1
        if order_type == 'HOLD':
            trades.loc[date]['Cash'] = cash
            continue
        if np.isnan(trades.loc[date][ticker]):
            trades.loc[date][ticker] = order_number * buy
        else:
            trades.loc[date][ticker] = order_number * buy + trades.loc[date][ticker]
        cash = cash - prices.loc[date][ticker] * order_number * buy - commission - impact * prices.loc[date][
            ticker] * order_number
        trades.loc[date]['Cash'] = cash

    holdings = trades.copy()
    temp = holdings.cumsum().fillna(method='ffill')
    temp = temp.fillna(0)
    temp['Cash'] = holdings['Cash']
    holdings = temp.fillna(method='ffill')
    portvals = holdings * prices

    return portvals
    total_value = portvals.sum(axis=1)
    return total_value