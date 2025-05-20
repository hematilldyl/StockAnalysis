import pandas as pd
import numpy as np
from util import get_data, plot_data
import datetime as dt

def author():
    return 'dhematillake3'

def testPolicy(symbol='JPM',sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=100000):
    data = get_data([symbol], pd.date_range(sd,ed))
    data['Difference']=data['JPM'].diff()
    data['Order'] = np.nan
    data['Shares'] = np.nan
    #will optimize later via df.shift()
    for i in range(0,len(data)-1):
        if i ==0:
            if data['Difference'].iloc[i+1]<0:
                data['Order'].iloc[i] = 'SELL'
                data['Shares'].iloc[i] = 1000
            else:
                data['Order'].iloc[i] = 'BUY'
                data['Shares'].iloc[i] = 1000
        if data['Difference'].iloc[i]<0 and data['Difference'].iloc[i+1]>0:
            data['Order'].iloc[i] = 'BUY'
            data['Shares'].iloc[i] = 2000
        if data['Difference'].iloc[i] > 0 and data['Difference'].iloc[i + 1] <0:
            data['Order'].iloc[i] = 'SELL'
            data['Shares'].iloc[i] = 2000
        orders = data.copy()
        orders = orders[['Order','Shares']]
    return orders


