import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import json
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
import requests

#Dobavljanje podataka sa API-ja sajta ispod
url = "https://alpha-vantage.p.rapidapi.com/query"

option = input("Which crypto do you want(eg. BTC, XRP, BNB, ETH...): ")
crypto_currency = option

querystring = {"market":"USD","symbol":crypto_currency,"function":"DIGITAL_CURRENCY_DAILY"}

headers = {
    'x-rapidapi-key': "bbc7016334msh345dccc59d546ebp16b0b5jsnfe4c16d39b90",
    'x-rapidapi-host': "alpha-vantage.p.rapidapi.com"
}

response = requests.request("GET", url, headers=headers, params=querystring)

#Pretvranje podataka u dataframe iz json formata
a_json = json.loads(response.text)
data = pd.DataFrame.from_dict(a_json['Time Series (Digital Currency Daily)'], orient="index")
data = data.reindex(index=data.index[::-1])


#U slucaju da api padne
#data = pd.read_csv('Bitso_BTCMXN_d.csv')

#Skaliranje podataka izmedju 0 i 1 da bi RNN lakse radio sa njima
scaler = MinMaxScaler(feature_range=(0, 1))

#Uzimanje samo potrebnih kolona tj. close kolone
scaled_data = scaler.fit_transform(data['4a. close (USD)'].values.reshape(-1, 1))
prediction_days = 100

#Pravljenje podataka nad kojima se vrsi treniranje
x_train, y_train = [], []

for x in range(prediction_days, len(scaled_data)):
    #Dodavanje sekvenci duzine 100
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])


x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Uzimanje poslednjih 20 posto za validacioni skup
x_validate, y_validate = np.array(x_train[:-round(len(x_train)*0.2)]), np.array(y_train[:-round(len(x_train)*0.2)])
x_validate = np.reshape(x_validate, (x_validate.shape[0], x_validate.shape[1], 1))

#Kreiranje modela i podesavanje odgovarajucih parametara
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

#Podesavanje optimajzera za nas RNN
model.compile(optimizer='adam', loss='mean_squared_error')

#Treniranje modela
model.fit(x_train, y_train, validation_data=(x_validate, y_validate), epochs=25, batch_size=32)


#Izdvajanje skupa podataka za test

test_data = data
#Prva zamisao nam je bila da uzmemo poslednjih 10 posto za testne podatke, ali
#ako ovo uradimo dobijamo gori rezultat nego inace
#test_data = data[:-round(len(x_train)*0.1)]

#U slucaju da api padne
#test_data = pd.read_csv('Bitso_BTCMXN_d.csv')
actual_prices = test_data['4a. close (USD)'].values

#Pretvarannje iz stringa u niz floatova
actual_prices1 = []
for price in actual_prices:
    actual_prices1.append([float(price)])
#print(actual_prices1)

total_dataset = pd.concat((data['4a. close (USD)'], test_data['4a. close (USD)']), axis=0)

#Ponavljanje postupka od gore samo sa testnim podacima
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.fit_transform(model_inputs)

x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Funkcija modela sa kojom prediktujemo vrednosti kriptovaluta
prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)
#print(prediction_prices)

#Izracunavanje vrednosti kriptovalute sutra
real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs) + 1, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print("Value of " + crypto_currency+" tommorow is: ", prediction[0][0])

#Iscrtavanje grafa koji sadrzi prave vrednosti i vrednosti koje je izracunao nas model
plt.plot(actual_prices1, color="blue", label="Actual prices")
plt.plot(prediction_prices, color="red", label="Prediction prices")
plt.title("Price prediction")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()

