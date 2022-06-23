# import fastai
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import math

# to plot within notebook
import matplotlib.pyplot as plt

# setting figure size
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 20, 10

# for normalizing data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))


def calculate_rms(pred):
    a = np.array(valid['Close']) - preds
    b = np.power(a, 2)
    c = np.nansum(b)
    d = len(b)
    mean = c / d
    return (math.sqrt(mean))


# read the file
path = "C:/Users/Karan Padariya/OneDrive/Desktop/study/Sem 5/ML/Assingment/SUNPHARMA.NS.csv"
df = pd.read_csv(path)

# print the head
print(df.head)

# setting index as date
df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
df.index = df['Date']

# plot
plt.figure(figsize=(16, 8))
plt.plot(df['Close'], label='Close Price history')
plt.show()

data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])

for i in range(0, len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

from fastai.tabular.all import add_datepart

add_datepart(new_data, 'Date')
new_data.drop('Elapsed', axis=1, inplace=True)  # elapsed will be the time stamp

new_data['mon_fri'] = 0
pd.options.mode.chained_assignment = None
for i in range(0, len(new_data)):
    if (new_data['Dayofweek'][i] == 0 or new_data['Dayofweek'][i] == 4):
        new_data['mon_fri'][i] = 1
    else:
        new_data['mon_fri'][i] = 0

# split into train and validation
a = int(len(df) - (len(df) * 20 / 100))
train = new_data[:a]
valid = new_data[a:]

x_train = train.drop('Close', axis=1)
y_train = train['Close']
x_valid = valid.drop('Close', axis=1)
y_valid = valid['Close']

from sklearn import neighbors
from sklearn.model_selection import GridSearchCV

x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)
x_valid_scaled = scaler.fit_transform(x_valid)
x_valid = pd.DataFrame(x_valid_scaled)

# using gridsearch to find the best parameter
params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)

y_train = y_train.fillna(1)
x_train = x_train.fillna(1)
# fit the model and make predictions
model.fit(x_train, y_train)
preds = model.predict(x_valid)

rms = calculate_rms(preds)
# plot
valid['Predictions'] = 0
valid['Predictions'] = preds
plt.plot(valid[['Close', 'Predictions']])
plt.plot(train['Close'])
plt.show()
