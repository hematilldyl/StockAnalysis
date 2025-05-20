import datetime as dt
  		  	   		  	  			  		 			     			  	 
import numpy as np  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import pandas as pd  		  	   		  	  			  		 			     			  	 
from util import get_data, plot_data  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
def author():
    return 'dhematillake3'
def compute_portvals(  		  	   		  	  			  		 			     			  	 
    orders_file="./orders/orders.csv",  		  	   		  	  			  		 			     			  	 
    start_val=1000000,  		  	   		  	  			  		 			     			  	 
    commission=9.95,  		  	   		  	  			  		 			     			  	 
    impact=0.005,  		  	   		  	  			  		 			     			  	 
):  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	  			  		 			     			  	 
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
    orders = pd.read_csv(orders_file,parse_dates=True,index_col='Date',na_values=['nan'])
    orders=orders.sort_index()
    start_date = orders.index[0]
    end_date = orders.index[-1]
    reference = get_data(['SPY'], pd.date_range(start_date, end_date))
    prices = np.zeros((reference.shape[0],len(orders['Symbol'].unique())+1))
    prices[:,-1]=1
    inc = 0
    for ticker in orders['Symbol'].unique():
        ticker_value = get_data([ticker],pd.date_range(start_date,end_date))
        prices[:,inc] = ticker_value[ticker].values
        inc+=1
    companies = orders['Symbol'].unique()
    prices = pd.DataFrame(index = reference.index,data=prices,columns=np.append(companies,'Cash'))
    trades = prices.copy()
    for col in trades.columns:
        trades[col]=np.nan

    cash = start_val
    for i in range(0,len(orders)):
        date = orders.index[i]
        ticker = orders['Symbol'].iloc[i]
        order_type = orders['Order'].iloc[i]
        order_number = orders['Shares'].iloc[i]
        buy = 1
        if order_type == 'SELL':
            buy = -1
        if np.isnan(trades.loc[date][ticker]):
            trades.loc[date][ticker] = order_number * buy
        else:
            trades.loc[date][ticker] = order_number * buy + trades.loc[date][ticker]
        cash = cash - prices.loc[date][ticker] * order_number * buy - commission - impact * prices.loc[date][ticker] * order_number
        trades.loc[date]['Cash'] = cash

    holdings = trades.copy()
    temp = holdings.cumsum().fillna(method='ffill')
    temp=temp.fillna(0)
    temp['Cash'] = holdings['Cash']
    holdings = temp.fillna(method='ffill')
    portvals = holdings*prices

    #return portvals
    total_value = portvals.sum(axis=1)
    return total_value
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
def test_code():  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    Helper function to test code  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    # this is a helper function you can use to test your code  		  	   		  	  			  		 			     			  	 
    # note that during autograding his function will not be called.  		  	   		  	  			  		 			     			  	 
    # Define input parameters  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    of = "./orders/orders-01.csv"
    #of = "./additional_orders/orders2.csv"
    sv = 1000000  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    # Process orders  		  	   		  	  			  		 			     			  	 
    portvals = compute_portvals(orders_file=of, start_val=sv,commission=0,impact=0)
    #if isinstance(portvals, pd.DataFrame):
    #    portvals = portvals[portvals.columns[0]]  # just get the first column
    #else:
    #    "warning, code did not return a DataFrame"
  		  	   		  	  			  		 			     			  	 
    # Get portfolio stats  		  	   		  	  			  		 			     			  	 
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = portvals.index[0]
    end_date = portvals.index[-1]
    opt_alloc_port = portvals
    opt_port_val = opt_alloc_port.sum(axis=1)
    daily_rets = (opt_port_val.iloc[1:] / opt_port_val.iloc[:-1].values) - 1
    avg_daily_ret = daily_rets.mean()
    std_daily_ret = daily_rets.std()
    cum_ret = (opt_port_val.iloc[-1] / opt_port_val.iloc[0]) - 1
    sharpe_ratio = np.sqrt(252) * avg_daily_ret / std_daily_ret

    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [  		  	   		  	  			  		 			     			  	 
        0.2,  		  	   		  	  			  		 			     			  	 
        0.01,  		  	   		  	  			  		 			     			  	 
        0.02,  		  	   		  	  			  		 			     			  	 
        1.5,  		  	   		  	  			  		 			     			  	 
    ]  		  	   		  	  			  		 			     			  	 
    '''		  	   		  	  			  		 			     			  	 
    # Compare portfolio against $SPX  		  	   		  	  			  		 			     			  	 
    print(f"Date Range: {start_date} to {end_date}")  		  	   		  	  			  		 			     			  	 
    print()  		  	   		  	  			  		 			     			  	 
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		  	   		  	  			  		 			     			  	 
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")  		  	   		  	  			  		 			     			  	 
    print()  		  	   		  	  			  		 			     			  	 
    print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		  	  			  		 			     			  	 
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")  		  	   		  	  			  		 			     			  	 
    print()  		  	   		  	  			  		 			     			  	 
    print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		  	  			  		 			     			  	 
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")  		  	   		  	  			  		 			     			  	 
    print()  		  	   		  	  			  		 			     			  	 
    print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		  	  			  		 			     			  	 
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")  		  	   		  	  			  		 			     			  	 
    print()  		  	   		  	  			  		 			     			  	 
    print(f"Final Portfolio Value: {portvals[-1]}")  		  	   		  	  			  		 			     			  	 
  	'''
  		  	   		  	  			  		 			     			  	 
if __name__ == "__main__":  		  	   		  	  			  		 			     			  	 
    test_code()  		  	   		  	  			  		 			     			  	 
