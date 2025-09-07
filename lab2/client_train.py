import grpc
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import os
os.environ["RAY_DEDUP_LOGS"] = "0"
import ray
ray.init()

import model_train_pb2 as pb2
import model_train_pb2_grpc as pb2_grpc

@ray.remote
class ClientActor:
    def __init__(self, client_id, client_type, train_data, test_data, scaler):
        self.client_id = client_id
        self.client_type = client_type

        self.train_data = train_data
        self.test_data = test_data

        self.scaler = scaler

        self.channel = grpc.insecure_channel("localhost:50051")
        self.stub = pb2_grpc.ModelTrainStub(self.channel)

    def send_train_data(self): 
        samples = []
        for i in range(len(self.train_data[0])):
            samples.append(pb2.sample(attrs=list(self.train_data[0][i]), label=int(self.train_data[1][i])))

        request = pb2.FitRequest(samples=samples)

        self.stub.Train(request)
        return
    
    def send_test_data(self):   
        samples = []
        for i in range(len(self.test_data[0])):
            samples.append(pb2.sample(attrs=list(self.test_data[0][i])))

        request = pb2.PredictRequest(samples=samples)

        response = self.stub.Predict(request)
        print(f"Client {self.client_id} - {self.client_type} - test accuray: ", accuracy_score(self.test_data[1], response.pred))
        return

def simulate_byzantine(client_type, X, y):
    if client_type == "normal":
        return X, y
    elif client_type == "noise_all":
        return X + np.random.normal(0, 2, X.shape), y
    elif client_type == "noise_one_feature":
        X_mod = X.copy()
        f = np.random.randint(X.shape[1])
        X_mod[:, f] += np.random.normal(0, 2, X.shape[0])
        return X_mod, y
    elif client_type == "flip_labels":
        y_mod = np.max(y) - y
        return X, y_mod
    else:
        raise ValueError(f"client type unknown: {client_type}")

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    iris = load_iris()
    X, y = iris.data, iris.target
    client_data_train = []
    client_data_test = []
    clients = []

    client_types = ["normal", "noise_all", "noise_one_feature", "flip_labels", "normal"]
    client_types = ["normal", "flip_labels", "normal", "normal", "normal"]
    # client_types = ["normal", "noise_all", "normal", "noise_one_feature", "normal"]

    for i in range(len(client_types)):
        # split dataset into the number of clients that will be simulated
        start_index = i * int(len(X)/len(client_types))
        end_index = (i + 1) * int(len(X)/len(client_types)) if i < 4 else len(X) # Ensure last part includes remaining data
        curr_X = X[start_index:end_index]
        curr_y = y[start_index:end_index]
        
        # modify the part of dataset according of the type of current client
        X_mod, y_mod = simulate_byzantine(client_types[i], curr_X, curr_y)

        scaler = StandardScaler()
        X_mod = scaler.fit_transform(X_mod)

        # split data into train and test
        X_train_part, X_test_part, y_train_part, y_test_part = train_test_split(
            X_mod, y_mod, test_size=0.2, random_state=42
        )

        # create clients
        clients.append(ClientActor.remote(i, client_types[i], train_data=(X_train_part, y_train_part), test_data=(X_test_part, y_test_part), scaler=scaler))

    # send training data
    ray.get([c.send_train_data.remote() for c in clients])

    # send test data
    results = ray.get([c.send_test_data.remote() for c in clients])