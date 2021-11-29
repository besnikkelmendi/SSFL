from array import array
import socket 
import os
import threading
from threading import Thread 
from socketserver import ThreadingMixIn 
import random
import tensorflow as tf
import h5py
import numpy as np

C = 1
MIN_NO_CLIENTS = 10

client_threads = []
client_ids = []
client_weights = []
server_weights_arr = []


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

def evaluate(parameters):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}

def getServerWeights():
    model.load_weights("main_model_weights.h5")

    # print(model.get_weights())
    return model.get_weights()

def createRandomSortedList(num, start = 1, end = 100):
        arr = []
        tmp = random.randint(start, end)
        
        for x in range(num):
            
            while tmp in arr:
                tmp = random.randint(start, end)
                
            arr.append(tmp)
            
        arr.sort()
        client_weights.clear()
        return arr
        
def FedAvg(c_weights, s_weights):
        sum = np.zeros((10,), dtype=int)
        print("Init shape: ", s_weights)
        for weights in c_weights:
            print("Weight: ", np.asarray(weights))
            sum = np.add(sum, np.asarray(weights))
        print("Sum: ", sum)
        avg_c_weights = []
        for weight in sum:
            avg_c_weights.append(float(float(weight)/float(len(c_weights))))
        
        s_weights[1] = np.asarray(avg_c_weights)
        print(s_weights)
        # ci
        return s_weights

def Average(s_weights):
    if len(client_weights) >= int(MIN_NO_CLIENTS*C):
        print("Averaging updates...\n")
        f_weights = FedAvg(client_weights, s_weights)
        client_weights.clear()
        # print("Final weights: \n", s_weights)
        print("Evauation initiated...\n")
        evaluation = evaluate(f_weights)
        print("Evaluation: \n",evaluation)
        # try:
        model.set_weights(s_weights)
        model.save("main_model_weights.h5")
        # except:
        #     print("An error occured while storing final parameters")
# Multithreaded Python server : TCP Server Socket Thread Pool
class ClientThread(Thread): 
    
    

    def __init__(self,conn,ip,port): 
        Thread.__init__(self) 
        self.conn = conn
        self.ip = ip 
        self.port = port 

        print( "New client is available - " + ip + ":" + str(port))
        
 
    def run(self):
        message = ""
        trained = False
        
        # print(str(message))
        self.conn.send(s_weights)
        # conn.send(b'Welcome to the Federated Learning Server!')
        while True:
            # try:
                data = self.conn.recv(1024)
                # print( "\nServer received data:", data)

                if  not trained:                
                    self.conn.send(b'Start training') 
                    trained = True
                else:
                    data = data.decode().replace("b'","")
                    data = data.replace("]",'')
                    data = data.replace("[",'')
                    data = data.split()
                    weights = []
                    for weight in data:
                        weights.append(float(weight))
                    print(weights)
                    client_weights.append(weights) 
                    print(len(client_weights))
                    Average(server_weights_arr) 
                    # FedAvg(s_weights, client_weights)
                    break
            

            # except:
            #     continue
            # print( "Server received data:", data)
            # data = conn.recv(1024)
        self.conn.close()
        # conn.close()

# Multithreaded Python server : TCP Server Socket Program Stub
TCP_IP = '0.0.0.0' 
TCP_PORT = 2004 
BUFFER_SIZE = 2048  # Usually 1024, but we need quick response 
s_weights = ""
with open("main_model_weights.h5","rb") as f:
              s_weights = f.read()

# h5f = h5py.File('main_model_weights.h5','r')
# print("H5 file: ", h5f.numpy())

server_weights_arr = getServerWeights()

tcpServer = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
tcpServer.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
tcpServer.bind((TCP_IP, TCP_PORT)) 
threads = [] 
 
while True: 
    # Average(server_weights_arr)
    tcpServer.listen(4) 
    print( "MNIST Federated Learning Server: Waiting for clients...\n" )
    (conn, (ip,port)) = tcpServer.accept() 
    newthread = ClientThread(conn,ip,port) 
    # client_threads.append(newthread)
    # if (len(threads)+1)>=2:
    # threads[0].start()
    threads.append(newthread)
    
    newthread.start()


    # if len(threads) >= MIN_NO_CLIENTS:
    #     print("Training initiated...\n")
    #     clients = createRandomSortedList(int(MIN_NO_CLIENTS*C), 0, len(threads)-2)
    #     # print(clients)
    #     for client in clients:
    #         threads[client].start()
    #         threads.pop(client)
    # client_ids.append(str(ip)+"|"+str(port))

    
                

 
for t in threads: 
    t.join() 