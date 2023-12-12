import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import torch 
import streamlit as st
import yfinance as yfin
import torch
import torch.nn as nn
yfin.pdr_override()

start = '2010-01-01'
end = '2023-12-10'


st.title("Stock Price Prediciton")
user_input = st.text_input("Enter Stock Ticker","AAPL")

df = data.data.get_data_yahoo(user_input,start,end)


st.subheader("Data Description")
st.write(df.describe())

df = df.reset_index()
st.subheader('Closing Price vs Time Chart')

fig1  = plt.figure(figsize=(12,6))
plt.plot(df['Date'],df.Close)
st.pyplot(fig1)


st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(12,6))
plt.plot(df['Date'],ma100,'r')
plt.plot(df['Date'],df.Close,'g')
st.pyplot(fig2)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7):])

from sklearn.preprocessing import MinMaxScaler
scaler  = MinMaxScaler(feature_range=(-1, 1))

data_training_array = scaler.fit_transform(data_training)

x_train = []
y_train = []
for i in range(100,data_training_array.shape[0]):
  x_train.append(data_training_array[i-100:i])
  y_train.append([data_training_array[i,0]])

x_train,y_train = np.array(x_train),np.array(y_train)
input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

model.load_state_dict(torch.load('model.pth'))

past_100_days = data_training.tail(100)
final_df = past_100_days._append(data_testing,ignore_index=True)
input_data = scaler.fit_transform(final_df)
x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append([input_data[i,0]])

x_test,y_test = np.array(x_test),np.array(y_test)
x_test = torch.from_numpy(x_test).float()
y_pred = model(x_test)

y_pred = scaler.inverse_transform(y_pred.detach())
y_test = scaler.inverse_transform(y_test)

st.subheader("Original vs Predicted Price")
fig3 = plt.figure(figsize=(12,6))
plt.plot(df['Date'][int(len(df)*0.7):],y_test,'b',label='Original Price')
plt.plot(df['Date'][int(len(df)*0.7):],y_pred,'g',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)
