import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import torch 
import streamlit as st
import yfinance as yfin
yfin.pdr_override()

start = '2010-01-01'
end = '2023-12-10'


st.title("Stock Price Prediciton")
user_input = st.text_input("Enter Stock Ticker","AAPL")

df = data.data.get_data_yahoo(user_input,start,end)

st.subheader('Data from 2010 - 2019') 
