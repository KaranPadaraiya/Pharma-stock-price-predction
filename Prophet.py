import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
    
    #to plot within notebook
import matplotlib.pyplot as plt
    
    #setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10
    
    #for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
   
def calculate_rms(preds):
    a=np.array(valid['Close'])-preds
    b=np.power(a,2)
    c=np.nansum(b)
    d=len(b)
    mean=c/d
    return(math.sqrt(mean))
    
    #read the file
df = pd.read_csv('ETH-USD.csv')
    
    #print the head
print(df.head)
    
    #setting index as date
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']
    
    #plot
plt.figure(figsize=(16,8))
plt.plot(df['Close'], label='Close Price history')
plt.show()
    
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
from fbprophet import Prophet
    
for i in range(0,len(data)):
     new_data['Date'][i] = data['Date'][i]
     new_data['Close'][i] = data['Close'][i]
    
new_data['Date'] = pd.to_datetime(new_data.Date,format='%Y-%m-%d')
new_data.index = new_data['Date']
    
    new_data.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)
    train = new_data[:1840]
    valid = new_data[1840:]
    
    #fit the model
    model = Prophet()
    model.fit(train)
    
    #predictions
    close_prices = model.make_future_dataframe(periods=len(valid))
    forecast = model.predict(close_prices)
    
    forecast_valid = forecast['yhat'][1840:]
    
    #plot
    valid['Predictions'] = 0
    valid['Predictions'] = forecast_valid.values
    
    plt.plot(train['y'])
    plt.plot(valid[['y', 'Predictions']])
    plt.show()
