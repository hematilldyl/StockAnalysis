  	  			  		 			     			  	 
import datetime as dt  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import numpy as np  		  	   		  	  			  		 			     			  	 
from scipy.optimize import minimize
import matplotlib.pyplot as plt  		  	   		  	  			  		 			     			  	 
import pandas as pd  		  	   		  	  			  		 			     			  	 
from util import get_data, plot_data  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 		  	   		  	  			  		 			     			  	 
def optimize_portfolio(  		  	   		  	  			  		 			     			  	 
    sd=dt.datetime(2008, 6, 1),
    ed=dt.datetime(2009, 6, 1),
    syms=["IBM", "X", "GLD", "JPM"],
    gen_plot=False,  		  	   		  	  			  		 			     			  	 
):  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		  	  			  		 			     			  	 
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		  	  			  		 			     			  	 
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		  	  			  		 			     			  	 
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		  	  			  		 			     			  	 
    statistics.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  	  			  		 			     			  	 
    :type sd: datetime  		  	   		  	  			  		 			     			  	 
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  	  			  		 			     			  	 
    :type ed: datetime  		  	   		  	  			  		 			     			  	 
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		  	  			  		 			     			  	 
        symbol in the data directory)  		  	   		  	  			  		 			     			  	 
    :type syms: list  		  	   		  	  			  		 			     			  	 
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		  	  			  		 			     			  	 
        code with gen_plot = False.  		  	   		  	  			  		 			     			  	 
    :type gen_plot: bool  		  	   		  	  			  		 			     			  	 
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		  	  			  		 			     			  	 
        standard deviation of daily returns, and Sharpe ratio  		  	   		  	  			  		 			     			  	 
    :rtype: tuple  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 


    # Read in adjusted closing prices for given symbols, date range  		  	   		  	  			  		 			     			  	 
    dates = pd.date_range(sd, ed)  		  	   		  	  			  		 			     			  	 
    prices_all = get_data(syms, dates)  # automatically adds SPY  		  	   		  	  			  		 			     			  	 
    prices = prices_all[syms]  # only portfolio symbols  		  	   		  	  			  		 			     			  	 
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    # find the allocations for the optimal portfolio  		  	   		  	  			  		 			     			  	 
    # note that the values here ARE NOT meant to be correct for a test case  		  	   		  	  			  		 			     			  	 
    allocs = np.ones(prices.shape[1])/prices.shape[1] #initial guess
    constraints = ({"type":"eq","fun": lambda x: np.sum(x)-1}) #alloc sum to 1
    bounds = tuple((0,1) for x in range(prices.shape[1])) #bound allocs 0 to 1
    portfolio_df = prices.copy()
    portfolio_df.fillna(method='ffill',inplace=True)
    portfolio_df.fillna(method='bfill', inplace=True)

    portfolio_df = portfolio_df/portfolio_df.iloc[0] #normalize
    def obj(allocs):
        allocs = np.array(allocs)
        allocated = portfolio_df*allocs
        portfolio_val = allocated.sum(axis=1)
        daily_rets = (portfolio_val.iloc[1:] / portfolio_val.iloc[:-1].values) - 1
        avg_daily_rets = daily_rets.mean()
        std_daily_rets = daily_rets.std()
        cum_ret = (portfolio_val.iloc[-1]/portfolio_val.iloc[0])-1
        sr = np.sqrt(252)*avg_daily_rets/std_daily_rets
        return -sr

    # add code here to find the allocations
    results = minimize(obj,allocs,method = "SLSQP",bounds = bounds, constraints = constraints)
    optimized_alloc = results.x
    # add code here to compute stats
    opt_alloc_port = optimized_alloc*portfolio_df
    opt_port_val = opt_alloc_port.sum(axis=1)
    daily_rets = (opt_port_val.iloc[1:] / opt_port_val.iloc[:-1].values) - 1
    avg_daily_rets = daily_rets.mean()
    std_daily_rets = daily_rets.std()
    cum_ret = (opt_port_val.iloc[-1] / opt_port_val.iloc[0]) - 1
    sharpe = np.sqrt(252) * avg_daily_rets / std_daily_rets


    cr, adr, sddr, sr = [  		  	   		  	  			  		 			     			  	 
        cum_ret,
        avg_daily_rets,
        std_daily_rets,
        sharpe,
    ]  #Cumulative return, Average Daily Return, Standard deviation of daily return, sharpe ratio


  		  	   		  	  			  		 			     			  	 
    # Get daily portfolio value  		  	   		  	  			  		 			     			  	 
    port_val = opt_port_val  # add code here to compute daily portfolio values
  		  	   		  	  			  		 			     			  	 
    # Compare daily portfolio value with SPY using a normalized plot  		  	   		  	  			  		 			     			  	 
    if gen_plot:  		  	   		  	  			  		 			     			  	 
        # add code to plot here  		  	   		  	  			  		 			     			  	 
        df_temp = pd.concat(  		  	   		  	  			  		 			     			  	 
            [port_val, prices_SPY/prices_SPY[0]], keys=["Portfolio", "SPY"], axis=1
        )  		  	   		  	  			  		 			     			  	 
        plot = df_temp.plot()
        plt.xlabel('Date')
        plt.ylabel('Normalized Price (Price_Day/Price_First_Date)')
        plt.title('Daily Normalized Portfolio Value vs S&P 500 Index SPY')
        plt.savefig('images/Figure1.png')
  		  	   		  	  			  		 			     			  	 
    return optimized_alloc, cr, adr, sddr, sr
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
def test_code():  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    This function WILL NOT be called by the auto grader.  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ["IBM", "X", "GLD", "JPM"]
  		  	   		  	  			  		 			     			  	 
    # Assess the portfolio  		  	   		  	  			  		 			     			  	 
    allocations, cr, adr, sddr, sr = optimize_portfolio(  		  	   		  	  			  		 			     			  	 
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True
    )  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    # Print statistics  		  	   		  	  			  		 			     			  	 
    print(f"Start Date: {start_date}")  		  	   		  	  			  		 			     			  	 
    print(f"End Date: {end_date}")  		  	   		  	  			  		 			     			  	 
    print(f"Symbols: {symbols}")  		  	   		  	  			  		 			     			  	 
    print(f"Allocations:{allocations}")  		  	   		  	  			  		 			     			  	 
    print(f"Sharpe Ratio: {sr}")  		  	   		  	  			  		 			     			  	 
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		  	  			  		 			     			  	 
    print(f"Average Daily Return: {adr}")  		  	   		  	  			  		 			     			  	 
    print(f"Cumulative Return: {cr}")  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
if __name__ == "__main__":  		  	   		  	  			  		 			     			  	 
    # This code WILL NOT be called by the auto grader  		  	   		  	  			  		 			     			  	 
    # Do not assume that it will be called  		  	   		  	  			  		 			     			  	 
    test_code()  		  	   		  	  			  		 			     			  	 
