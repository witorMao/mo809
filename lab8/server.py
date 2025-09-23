import splitlearning_pb2 as pb2
import splitlearning_pb2_grpc as pb2_grpc
import grpc
from concurrent import futures
import tensorflow as tf
import keras
from keras.models import Model
import time
import pandas as pd
import os

class ServerModel(tf.keras.models.Model):
    def __init__(self):
        super(ServerModel, self).__init__()
        self.dense1 = keras.layers.Dense(64, activation='relu')     # Camada intermediária
        self.dense2 = keras.layers.Dense(10, activation='softmax')  # Camada de saída 

    def call(self, input):
        x = self.dense1(input)
        x = self.dense2(x)
        return x
    

class SplitLearningService(pb2_grpc.SplitLearningServicer):

    def __init__(self):
        self.server_model = create_server_model((128,))
        self.optimizer    = tf.keras.optimizers.Adam()
        self.metrics      = tf.keras.metrics.SparseCategoricalAccuracy()

def SendClientActivations(self, request, context):
    activations = tf.convert_to_tensor(request.activations, dtype=tf.float32)
    # -1 faz com que a dimensao seja calculada automaticamente
    activations = tf.reshape(activations, (request.batch_size, -1)) 
    labels      = tf.convert_to_tensor(request.labels, dtype=tf.float32)
    print(activations.shape)

    global epoch

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(activations)
        predictions = self.server_model(activations, training=True)
        loss        = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
        loss        = tf.reduce_mean(loss)
        acc         = self.metrics(labels, predictions)

    print("BACKWARD")
    server_gradients = tape.gradient(loss, self.server_model.trainable_variables)
    self.optimizer.apply_gradients(zip(server_gradients, self.server_model.trainable_variables))

    activations_gradients = tape.gradient(loss, activations)
    response              = pb2.ServerToClient()

    response.gradients.extend(activations_gradients.numpy().flatten())
    response.loss = loss.numpy()
    response.acc  = acc.numpy()
    epoch        += 1

    print(f"Epoch {epoch} - Loss: {loss.numpy()} - Acc: {acc.numpy()}")
    return response

def serve():
    global server
    MAX_MESSAGE_LENGTH = 20 * 1024 * 1024 * 10
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ])
    pb2_grpc.add_SplitLearningServicer_to_server(SplitLearningService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started on port 50051")
    server.wait_for_termination()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('results.csv', names=['epoch', 'batch', 'loss', 'accuracy', 'latencia', 'tx', 'rx'])

fig, ax = plt.subplots(2, 2, figsize=(15, 7.5))
ax      = ax.flatten()

sns.lineplot(x=df.index, y=df.loss, color='b', ax=ax[0], linewidth=2)
sns.lineplot(x=df.index, y=df.accuracy, color='k', ax=ax[1], linewidth=2)
sns.lineplot(x=df.index, y=df.latencia, color='r', ax=ax[2], linewidth=2)
sns.lineplot(x=df.index, y=df.tx.cumsum(), color='orange', ax=ax[3], linewidth=2)

for _ in range(4):
    ax[_].set_xlabel('Epochs (#)', size=13)
    ax[_].grid(True, linestyle=':')
    ax[_].set_xticks((range(0, len(df), 781)), ('1', '2', '3', '4', '5', ' 6'))

ax[0].set_ylabel('Loss', size=13)
ax[1].set_ylabel('Accurácia (%)', size=13)
ax[2].set_ylabel('Latência (s)', size=13)
ax[3].set_ylabel('Bytes Transmited (MB)', size=13)
ax[3].set_yscale('log')