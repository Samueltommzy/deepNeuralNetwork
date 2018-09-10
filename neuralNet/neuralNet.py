from numpy import  exp, array,dot,random
import numpy as np

class neuralNetwork():
    def __init__(self):
        random.seed(1)
        self.weights = 2*random.random((3,1)) -1
        self.weight2 = 2*random.random((4,1)) - 1
    def __sigmoid(self,x):
        return 1/(1 + exp(-x))
    def __sigmoid_derivative(self,x):
        return  x *(1-x)
    def think(self,inputs):
        l1 =  self.__sigmoid(dot(inputs,self.weights))
        l2 = self.__sigmoid(dot(l1,self.weight2))
        return l2
    def train_model(self,train_input,train_output,iterations):
        for iteraton in range(iterations):
            output = self.think(train_input)
            error = train_output - output
            if iteraton % 1000 == 0:
                print("Error : " , str(np.mean(np.abs(error)) ))
            adjustment  = dot(train_input.T , error* self.__sigmoid_derivative(output))
            self.weights += adjustment
if ( __name__  == '__main__'):
    neuralnet = neuralNetwork()
    print('Random starting weights :')
    print(neuralnet.weights)
    train_set = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    output_set = array([[0,1,1,0]]).T
    neuralnet.train_model(train_set,output_set,10000)
    print('New weight after training: ')
    print(neuralnet.weights)
    print('considering new input [0,0,0] -> ?: ')
    print(neuralnet.think(array([[0,0,0]])))
        
