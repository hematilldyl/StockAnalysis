import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web
from pandas import concat
from scipy.stats import norm
import datetime as dt
import random

#LSTM
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation



style.use('ggplot')

'''
Author: Dylan Hematillake
Date: 03/01/2018

Function: monte_carlo_portfolio

Parameters: 
	mean of returns as $/time unit
	stdev of returns as $/time unit
	IV (intial value/principal) as $
	yearly_contributions (can be any time unit) $/time unit
	time (whichever time unit) time unit
Returns: 
	Plot of portfolio performance in as an example and the total 
	returns over the N trials of simulation
Purpose:	
	Take mean return and stdev as percentages
	the initial investment, the yearly contributions
	along with the timeline of interest to generate Monte Carlo
	simulation of the portfolio as a whole. Works quite nicely for
	index portfolios. Can take other timelines, monthly for example
	only if the appropriate mean, stdev are used then
	
Imports and Libraries Used:
	Pandas, Numpy, Matplotlib, Random, Scipy
'''

def monte_carlo_portfolio(mean,stdev,IV,yearly_contribution,time):
    #records
    historicROI=[]
    Returns=[]
    Profit=[]
    #run the simulation N times (set to 100 here)
    for k in range(0,100):
	#temp records	
        ROI=[]
        Year=[]
	
	#run through the timeline of interest in the time units of the mean and stdev 
        for i in range(0,time):
            Year.append(i+1)
            random_return = random.SystemRandom()
            year_return=norm.ppf(random_return.random(),loc=mean,scale=stdev)/100
            if i == 0:
                investment = (IV*(1+year_return))+yearly_contribution
                ROI.append(investment)
            else:
                investment = (ROI[i-1]*(1+year_return))+yearly_contribution
                ROI.append(investment)
        Returns.append(ROI[len(ROI)-1])
        historicROI.append(ROI)
        Profit.append(ROI[len(ROI)-1]-(yearly_contribution*time))
    fig,axes = plt.subplots(2,1)
    plt.tight_layout(pad=1.08,h_pad=5)
    axes[0].set_title("Trend of Portfolio in Trial 1")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Value of Investment ($)")
    axes[0].scatter(Year,historicROI[2])
    x=np.linspace(1,100,num=100)
    axes[1].set_title("Return per Trial")
    axes[1].set_xlabel("Trial #")
    axes[1].set_ylabel("Value of Investment ($)")
    axes[1].bar(x,Profit)
    plt.show()

#tangerine balanced growth fund    
#monte_carlo(5.23,9.04,1000,1500,40) 


'''
Author: Dylan Hematillake
Date: 03/3/2018

Function: monte_carlo_stockForecast

Parameters: 
	stock: a string with the appropriate stock ticker
	
Returns: 
	Plot of the stock for N number of simulations,
	suggested no more than 5
	
Purpose:	
	To perform Monte Carlo simulation on a desired stock
	
User Defined Dependencies:
	get_data(). A repeatable function to get the stock
	data using pandas web DataReader. 
	
Imports and Libraries Used:
	Pandas, Numpy, Matplotlib, Datetime, Random, Scipy
'''

#fetch some stock data from google
def get_data(stock):
    start = dt.datetime(2000,1,1)
    end = dt.datetime(2017,12,31)
    df= web.DataReader(stock,'google',start,end)
    return df

#take a stock name and product a few monte carlo forecasts
#TP=YS*exp(r), drift + random to get random walk, ln(St/St-1)=alpha+Zsigma
def monte_carlo_stockForecast(stock):
    #get the data
    df=get_data(stock)
    #format the data
    YS = df['Close'].values
    YS_forecast = df['Close'].values
    length = len(YS_forecast)
    print(YS_forecast)
    df['Close Lag']=np.log(df['Close'].shift(1)/df['Close'])
    DR = df['Close Lag']
    
    #calculate the statistics
    mean=np.mean(DR)
    variance=np.var(DR)
    stdev=np.std(DR)
    drift = mean-((variance)/2)
    
    print(mean)
    print(variance)
    print(stdev)
    print(drift)
	
    col = ['g','m','r','b']

    #plot 4 trials
    for j in range(0,4):
        for i in range(0,50):
	    #calculate 50 days for forecast	
            rate = np.exp(drift+(stdev*norm.ppf(random.random())))
            index = len(YS_forecast)-1
            FS = [YS_forecast[index]*rate]
            YS_forecast = np.append(YS_forecast,FS) 
    
        plt.plot(YS_forecast,color=col[j],label="Forecast "+str(j))
        YS_forecast = YS_forecast[:length]
    plt.plot(YS,color="k",label="Actual")
    plt.legend()
    plt.show()
    
#monte_carlo_stockForecast('googl')    

'''
Author: Dylan Hematillake
Date: 03/3/2018

Function: stock_fit_LSTM

Parameters: 
	stock: a string with the appropriate stock ticker
	
Returns: 
	Plot of the stock with a predicted model from a
	4 layer LSTM with dropout 0.4, 0.6 and activation
	selu built with build_model.
	
Purpose:	
	To develop a behaviour model for the stock trend
	using an LSTM.
	
User Defined Dependencies:
	get_data(): A function to get the stock data using 
	pandas web DataReader. Takes a stock string.
	series_to_supervised(): A function to modfy the input data
	to lag it in order to have proper supervised learning. Only
	requires the input data.
	split_data(): Splits the data into training and testing
	sets. Input is the data.
	differencing(): Take a time series and performs first order
	log differencing to remove trends, seasonality and rolling variance
	idifferencing(): Undoes differencing(), requires the original data and 
	the differenced data
	build_model(): Takes the training features and develops the LSTM.
	
Imports and Libraries Used:
	Pandas, Matplotlib, Numpy, Keras
 '''
	
#frames data as supervised learning 
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

#split the data into training testing sets and reshape it
#to sample=sample.size,time_step=1,features=NumCols
def split_data(data):
    data=data.values
    train_length = int(np.floor(0.8*len(data)))
    train,test = data[0:train_length,:],data[train_length:,:]
    #samples = len, time-step, features
    train_X, train_y = train[:,:-1],train[:,-1]
    test_X,test_y=test[:,:-1],test[:,-1]
    train_X = train_X.reshape(train_X.shape[0],1,train_X.shape[1])
    test_X = test_X.reshape(test_X.shape[0],1,test_X.shape[1])
    print(train_X.shape,train_y.shape,test_X.shape,test_y.shape)
    return train_X,train_y,test_X,test_y,train_length

#log differencing 
def differencing(data):
    differenced = []
    for i in range(1,len(data)):
        #differenced.append(np.log(data.iloc[i])-np.log(data.iloc[i-1]))
        differenced.append((data.iloc[i])-(data.iloc[i-1]))
    return differenced

#reverse the log differencing
def idifferencing(data,differenced_data):
    undifferenced = [data[0]]
    for i in range(1,len(data)-1):
        #undifferenced.append(np.exp(differenced_data[i])+data[i-1])
        undifferenced.append((differenced_data[i])+data[i-1])
    return undifferenced

#build the model with LSTM recurrent layer 50,100,10 units with dropout, one output (predictor) 5 features loaded, 1 time step
#n number of samples
def build_model(train_X):
    model = Sequential()
    model.add(LSTM(100,input_shape=(train_X.shape[1],train_X.shape[2]),return_sequences=True,dropout=0.4))
    model.add(LSTM(50,return_sequences=True,dropout=0.6))
    model.add(LSTM(10,dropout=0.4))
    model.add(Dense(1,activation='selu'))
    model.compile(optimizer='adam',loss='mae')
    return model

#LSTM Model on Stock      
def stock_fit_LSTM(stock):
   df=get_data(stock)
   historic=(df['Open'].copy()).values
  
   for i in range(0,5):
       df.iloc[1:,i]=differencing(df.iloc[:,i])
       df.iloc[0,i]=0

   data=series_to_supervised(df,1,1)
   data.drop(data.columns[[6,7,8,9]],axis=1,inplace=True)
   train_X,train_y,test_X,test_y,train_length=split_data(data)
   model=build_model(train_X)
   history = model.fit(train_X,train_y,batch_size=64,epochs=2,validation_data=(test_X,test_y),verbose=2)

   estimator = model.predict(test_X)
   
   historic=historic[train_length:]
   

   iestimator=idifferencing(historic,estimator)
   iestimator=pd.Series(iestimator)
   iestimator=iestimator.reshape((len(iestimator),1))

   test_y=idifferencing(historic,test_y)
   test_y=pd.Series(test_y)
   test_y=test_y.reshape((len(test_y),1))


   print(test_y.shape)
   plt.plot(iestimator,color="r",label="Predicted")
   plt.plot(test_y,color="k",label="Actual")
   plt.legend()
   plt.show()

   
