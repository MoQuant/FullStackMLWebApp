# ENTER FMP KEY IN THIS FUNCTION
def auth():
	key = ''
	return key


# URL + Endpoint
url = 'https://financialmodelingprep.com/'
endpoint = 'api/v3/historical-price-full/{}?apikey={}'

import asyncio
import websockets
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import requests

def StockData(ticker):
	URL = url + endpoint.format(ticker, auth())
	resp = requests.get(URL).json()
	df = pd.DataFrame(resp['historical'])
	return df['adjClose'].values.tolist()[::-1]


class Model(nn.Module):

    def __init__(self, window, output):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(window, 250)
        self.layer2 = nn.Linear(250, 100)
        self.layer3 = nn.Linear(100, output)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x

def AddMetrics(x, days=50):
    dataset = []
    n = len(x)
    for i in range(days, n):
        box = x[i-days:i]
        ma = np.mean(box)
        sd = np.std(box)
        low = ma - 2*sd
        up = ma + 2*sd
        dataset.append([x[i], ma, low, up])
    return pd.DataFrame(dataset, columns=['Price','MA','LB','UB'])

def TrainTest(x, prop=0.85):
    I = int(prop*len(x))
    train = x[:I]
    test = x[I:]
    return train, test
    
def Inputs(dataset, window=100, output=30):
    n = len(dataset)
    training_data = []
    for w in range(window, n-output+1):
        a1, a2, a3, a4 = np.array(dataset[w-window:w]).T.tolist()
        b1, b2, b3, b4 = np.array(dataset[w:w+output]).T.tolist()
        training_data.append([a1 + a2 + a3 + a4, b1])
    IN = [torch.tensor(item[0], dtype=torch.float32) for item in training_data]
    OUT = [torch.tensor(item[1], dtype=torch.float32) for item in training_data]
    return torch.stack(IN), torch.stack(OUT)

def Outputs(dataset, window):
    a1, a2, a3, a4 = np.array(dataset[-window:]).T.tolist()
    X = torch.tensor(a1 + a2 + a3 + a4, dtype=torch.float32)
    return torch.stack((X,)), a1

async def Server(ws, path):
	print('Connected to client.........')
	while True:
		# HANDLES INPUT DATA
		resp = await ws.recv()
		input_params = json.loads(resp)
		ticker = input_params[0]
		epochs = int(input_params[1])
		window = int(input_params[2])
		output = int(input_params[3])
		lr = float(input_params[4])
		lookback = int(input_params[5])
		prop = float(input_params[6])

		msg = {'type':'update', 'payload': 'Inputs processed'}
		await ws.send(json.dumps(msg))

		# BUILD DATASET
		close = StockData(ticker)
		df = AddMetrics(close)
		model = Model(int(window*4), output)
		train, test = TrainTest(df)

		X, Y = Inputs(train, window=window, output=output)

		msg = {'type':'update', 'payload': 'Dataframe Built'}
		await ws.send(json.dumps(msg))

		criterion = nn.MSELoss()
		optimizer = optim.Adam(model.parameters(), lr=lr)

		msg = {'type':'update', 'payload': 'Training Model'}
		await ws.send(json.dumps(msg))

		for epoch in range(epochs):
			outputs = model(X)
			loss = criterion(outputs, Y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if (epoch + 1) % 100 == 0:
				#print('Epochs left: ', epochs-epoch-1)
				msg = {'type':'update', 'payload': f'Epochs left: {epochs - epoch - 1}'}
				await ws.send(json.dumps(msg))

		msg = {'type':'update', 'payload': 'Training Completed'}
		await ws.send(json.dumps(msg))

		XX, history = Outputs(test, window)

		with torch.no_grad():
			test_outputs = model(XX)

		predictions = test_outputs[-1].numpy().tolist()

		xa = list(range(len(history)))
		xb = list(range(len(history), len(history) + len(predictions)))


		msg = {'type':'data', 'payload': [xa, xb, history, predictions]}
		await ws.send(json.dumps(msg))

loop = asyncio.get_event_loop()
loop.run_until_complete(websockets.serve(Server, 'localhost', 8080))
loop.run_forever()

