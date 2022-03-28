import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
from plotly import graph_objs as go

start = '2010-01-01'
end = '2021-12-31'


st.title("Stock Trend Prediction")

user_input = st.text_input("Enter Stock Ticker", 'AAPl')
df = data.DataReader(user_input, 'yahoo', start, end)
data_test_visual = pd.DataFrame(df[int(len(df)*0.70):int(len(df))])

#Describing Data

st.subheader('Data from 2010 - 2021')
st.write(df.describe())

df = df.reset_index()

#Visualisation
st.subheader('Closing Price vs Time Chart')
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Stock_Close"))
fig.layout.update(xaxis_rangeslider_visible=True,autosize=False,width=800,height=550)
st.plotly_chart(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Stock_Close"))
fig.add_trace(go.Scatter(x=df['Date'], y=ma100, name="MA100"))
fig.layout.update(xaxis_rangeslider_visible=True,autosize=False,width=800,height=550)
st.plotly_chart(fig)

st.subheader('Closing Price vs Time Chart with 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="stock_close"))
fig.add_trace(go.Scatter(x=df['Date'], y=ma100, name="MA100"))
fig.add_trace(go.Scatter(x=df['Date'], y=ma200, name="MA200"))
fig.layout.update(xaxis_rangeslider_visible=True,autosize=False,width=800,height=550)
st.plotly_chart(fig)

#splitting data into Train & Tests

data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_train_array = scaler.fit_transform(data_train)



#Load my ML Model

model = load_model('keras_model.h5')

#Testing Part

past_100_days = data_train.tail(100)
final_df = past_100_days.append(data_test, ignore_index= True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range (100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test , y_test = np.array(x_test), np.array(y_test)


# Making Predictions

y_predicted = model.predict(x_test)

scale_factor = float(1/scaler.scale_)
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor


# Final Graph
data_test_visual["Original"] = y_test
data_test_visual["Predicted"] = y_predicted

data_test_visual = data_test_visual.reset_index()

st.subheader('Original Price vs Final Price')
fig = go.Figure()
fig.add_trace(go.Scatter(x=data_test_visual['Date'], y=data_test_visual['Predicted'], name="Predicted Price"))
fig.add_trace(go.Scatter(x=data_test_visual['Date'], y=data_test_visual['Original'], name="Original Price"))
fig.layout.update(xaxis_rangeslider_visible=True,autosize=False,width=800,height=550)
st.plotly_chart(fig)
