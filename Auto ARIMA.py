# import fastai
# import pmdarima
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import math

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
path=("C:/Users/Karan Padariya/OneDrive/Desktop/study/Sem 5/ML/Assingment/BIOCON.NS.csv")
df = pd.read_csv(path)

#print the head
print(df.head)

#setting index as date
df['Date'] = pd.to_datetime(df.Date,format='%d-%m-%Y')
df.index = df['Date']

#plot
plt.figure(figsize=(16,8))
plt.plot(df['Close'], label='Close Price history')
plt.show()

data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

for i in range(0,len(data)):
     new_data['Date'][i] = data['Date'][i]
     new_data['Close'][i] = data['Close'][i]


from fastai.tabular.all import add_datepart
add_datepart(new_data, 'Date')
new_data.drop('Elapsed', axis=1, inplace=True)  #elapsed will be the time stamp

new_data['mon_fri'] = 0
pd.options.mode.chained_assignment = None
for i in range(0,len(new_data)):
    if (new_data['Dayofweek'][i] == 0 or new_data['Dayofweek'][i] == 4):
        new_data['mon_fri'][i] = 1
    else:
        new_data['mon_fri'][i] = 0

#split into train and validation
a=int(len(df)-(len(df)*20/100))
train = new_data[:a]
valid = new_data[a:]


from pmdarima.arima import auto_arima

training = train['Close']
validation = valid['Close']
training=training.fillna(1)
validation=validation.fillna(1)
model = auto_arima(training, start_p=1, start_q=1,max_p=3, max_q=3, m=12,start_P=0, seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)


model.fit(training)

forecast = model.predict(n_periods=460)
forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])

rmas=calculate_rms(np.array(forecast['Prediction']))

#plot
plt.plot(train['Close'])
plt.plot(valid['Close'])
plt.plot(forecast['Prediction'])
plt.plot()
