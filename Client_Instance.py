import Client
import socket
import tensorflow as tf
import ssl

HOST = socket.gethostname()
PORT = 2004


ssl._create_default_https_context = ssl._create_unverified_context

# Load and compile Keras model
model =  model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(10, tf.nn.softmax, input_shape=(784,),
                            kernel_initializer='zeros')
])

model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

# Load MNIST dataset

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Flattening
x_train = x_train.reshape(60000,-1)
x_test = x_test.reshape(10000, -1)

class CifarClient():
    def get_parameters(self):
        model.load_weights("client_weights.h5")
        return model.get_weights()

    def fit(self, parameters):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=15, batch_size=32, steps_per_epoch=5)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}

    def get_final_parameters(self):
        return model.get_weights()

# Start Flower client
#fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=CifarClient())


client = Client.Client(PORT,HOST,CifarClient())
client.start_client()