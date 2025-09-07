import grpc
from concurrent import futures
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

import model_train_pb2 as pb2
import model_train_pb2_grpc as pb2_grpc

class ModelTrainGrpcServer(pb2_grpc.ModelTrainServicer):
    def __init__(self, expected_clients=5):
        self.client_data = []
        self.model_all = KNeighborsClassifier()
        self.model_filtered = KNeighborsClassifier()
        self.expected_clients = expected_clients
        self.received_clients = 0

    def Train(self, request, context):
        X = np.array([s.attrs for s in request.samples])
        y = np.array([s.label for s in request.samples])

        self.client_data.append((X, y))
        self.received_clients += 1
        print(f"[Server] received {X.shape[0]} samples (Client {self.received_clients}/{self.expected_clients})")

        # start training
        if self.received_clients == self.expected_clients:
            print("\n[Server] Received data from all clients. Starting training...")
            self.TrainModels()
            self.received_clients = 0
            self.client_data = []

        return pb2.FitResponse(accuracy=0.0)

    def Predict(self, request, context):
        X_test = np.array([s.attrs for s in request.samples])
        preds = self.model_filtered.predict(X_test).tolist()
        return pb2.PredictResponse(pred=preds)

    # def DetectByzantine(self, threshold=2.0):
    #     stats = [np.mean(X, axis=0) for X, _ in self.client_data]
    #     global_mean = np.mean(stats, axis=0)
    #     std_dev = np.std(stats, axis=0)

    #     normal_clients = []
    #     for idx, (X, y) in enumerate(self.client_data):
    #         z_score = np.abs(stats[idx] - global_mean) / (std_dev + 1e-8)
    #         if np.all(z_score < threshold):
    #             normal_clients.append((X, y))
    #         else:
    #             print(f"[Server] Client {idx} is bizantine")
    #     return normal_clients

    def DetectByzantine(self, threshold=1.8):
        """
        Detects Byzantine clients based on normalized features and labels.
        Clients with feature/label means too far from global mean are considered Byzantine.
        """
        normal_clients = []

        # compute combined feature + label mean for each client
        stats = []
        for X, y in self.client_data:
            feature_mean = np.mean(X, axis=0)
            label_mean = np.mean(y)
            stats.append(np.concatenate([feature_mean, [label_mean]]))

        stats = np.array(stats)
        global_mean = np.mean(stats, axis=0)
        std_dev = np.std(stats, axis=0) + 1e-8  # avoid division by zero

        for idx, (X, y) in enumerate(self.client_data):
            z_score = np.abs(stats[idx] - global_mean) / std_dev
            # if any feature or label is too far from global mean -> Byzantine
            if np.any(z_score > threshold):
                print(f"[Server] Client {idx} detected as Byzantine (z-scores: {z_score})")
            else:
                normal_clients.append((X, y))

        return normal_clients


    def TrainModels(self):
        # Treino sem filtragem
        X_all = np.vstack([X for X, _ in self.client_data])
        y_all = np.hstack([y for _, y in self.client_data])
        self.model_all.fit(X_all, y_all)
        acc_all = self.model_all.score(X_all, y_all)
        print(f"[Server] Train without bizantine detection - Accuracy: {acc_all:.4f}")

        # Treino com filtragem
        normal_clients = self.DetectByzantine()
        if len(normal_clients) > 0:
            X_f = np.vstack([X for X, _ in normal_clients])
            y_f = np.hstack([y for _, y in normal_clients])
            self.model_filtered.fit(X_f, y_f)
            acc_f = self.model_filtered.score(X_f, y_f)
            print(f"[Server] Train with bizantine detection - Accuracy: {acc_f:.4f}")
        else:
            print("[Server] All clients detected as bizantine!")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = ModelTrainGrpcServer(expected_clients=5)
    pb2_grpc.add_ModelTrainServicer_to_server(servicer, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("[Server] gRPC server running on port 50051")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()