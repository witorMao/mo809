import grpc
import model_train_pb2 as pb2
import model_train_pb2_grpc as pb2_grpc
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class ModelTrainGrpcClient:
    def __init__(self, host='localhost', port=50051):
        self.channel = grpc.insecure_channel(f"{host}:{port}")
        self.stub = pb2_grpc.ModelTrainStub(self.channel)

    def train(self, x_train, y_train):
        samples = [
            pb2.sample(attrs=list(attrs), label=int(label))
            for attrs, label in zip(x_train, y_train)
        ]
        request = pb2.FitRequest(samples=samples)
        return self.stub.Train(request).accuracy

    def predict(self, x_test):
        samples = [pb2.sample(attrs=list(attrs)) for attrs in x_test]
        request = pb2.PredictRequest(samples=samples)
        response = self.stub.Predict(request)
        return response.pred

if __name__ == "__main__":
    # Load data
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    client = ModelTrainGrpcClient()

    # Train model
    train_acc = client.train(x_train, y_train)
    print(f"[Client] Training accuracy: {train_acc:.4f}")

    # Predict
    predictions = client.predict(x_test)
    test_acc = accuracy_score(y_test, predictions)
    print(f"[Client] Test accuracy: {test_acc:.4f}")
