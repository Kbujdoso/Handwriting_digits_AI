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
            ]  #felbontja a listát megfelelő nagysárúra ebben az esetben az adat tömbünket
            for mini_batch in mini_batches: 
                self.update_mini_batch(mini_batch, eta)
            if test_data: 
                print ("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else: 
                print ("Epoch {0} complete".format(j))

#gradians
    def update_mini_batch(self, mini_batch, eta): 
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch: 
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights,nabla_w)]
        self.biases = [b - (eta/len(mini_batch)) * nb
                       for b, nb in zip(self.weights, nabla_b)]


#backprop

    def backprop(self, x, y): 
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights): 
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y * sigmoid_prime(zs[-1]))
        nabla_b[-1] = delta 
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for i in range(2, self.num_layers): 
            z = zs[-i]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-1+1].transpose(),delta) * sp
            nabla_b[-i] = delta 
            nabla_w[-i] = np.dot(delta, activations[-i-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)),y) 
        for x ,y in test_data]
        return sum(int(x==y) for (x, y) in test_results)
    
    def cost_derivative(self, output_activations,y):
        return (output_activations-y)






def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    sigmoid(z)*(1-sigmoid(z))
    