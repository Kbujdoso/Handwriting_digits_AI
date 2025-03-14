import numpy as np
import random
class Network(object):
    def __init__(self, sizes):
        self.num_layers =  len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(x,y) 
                        for x,y in zip(sizes[:-1], sizes[1:])]
        
    
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a)+ b)
    

    #feldarabolja az inputokat kis csoportokra és mindegyikre elvégzik a műveleteket
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: 
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs): 
            random.shuffle(training_data) #randomizálja a kódot
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]  #felbontja a listát megfelelő angysárúra ebben az esetben az adat tömbünket
            for mini_batch in mini_batches: 
                self.update_mini_batch(mini_batch, eta)
            if test_data: 
                print ("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else: 
                print ("Epoch {0} complete"-format(j))


    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def update_mini_batch(self, mini_batch, eta): 
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch: 
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            