import grpc
from concurrent import futures
import model_train_pb2 as pb2
import model_train_pb2_grpc as pb2_grpc
from sklearn.neighbors import KNeighborsClassifier

class ModelTrainGrpcServer(pb2_grpc.ModelTrainServicer):
    def __init__(self):
        self.model = KNeighborsClassifier()

    def Train(self, request, context):
        # Extract features and labels
        x_train = [list(sample.attrs) for sample in request.samples]
        y_train = [sample.label for sample in request.samples]

        if len(x_train) == 0:
            context.set_details("No training samples provided")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return pb2.FitResponse()

        print(f"[Server] Received {len(x_train)} training samples")

        # Train model
        self.model.fit(x_train, y_train)
        acc = float(self.model.score(x_train, y_train))  # cast to Python float
        return pb2.FitResponse(accuracy=acc)

    def Predict(self, request, context):
        x_test = [list(sample.attrs) for sample in request.samples]

        if len(x_test) == 0:
            context.set_details("No prediction samples provided")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return pb2.PredictResponse()

        preds = self.model.predict(x_test)
        preds_list = [int(p) for p in preds]  # cast to Python int
        return pb2.PredictResponse(pred=preds_list)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_ModelTrainServicer_to_server(ModelTrainGrpcServer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("[Server] gRPC server started on port 50051")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
