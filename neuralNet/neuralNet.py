from numpy import  exp, array, dot, random, abs, mean

class neuralNetwork():
    def __init__(self):
        random.seed(1)
        self.weights = 2*random.random((3, 4)) - 1
        self.weight2 = 2*random.random((4, 1)) - 1
    def __sigmoid(self, x):
        return 1/(1 + exp(-x))
    def __sigmoid_derivative(self, x):
        return  x *(1-x)
    def __scaled_layer1(self,x):
        return self.__sigmoid(dot(x,self.weights))
    def think(self, inputs):
        l1 =  self.__sigmoid(dot(inputs, self.weights))
        l2 = self.__sigmoid(dot(l1, self.weight2))
        return l2
    def train_model(self, train_input, train_output, iterations):
        for iteraton in range(iterations):
            output = self.think(train_input)
            error = train_output - output
            if iteraton % 1000 == 0:
                print("Error : " , str(mean(abs(error)) ))
            layer2_Delta = error * self.__sigmoid_derivative(self.think(train_input))
            layer1_error = dot(layer2_Delta,self.weight2.T)
            layer1_Delta = layer1_error * self.__sigmoid_derivative(self.__scaled_layer1(train_input))
            self.weights += dot(train_input.T,layer1_Delta.T)
            self.weight2 += dot(self.__scaled_layer1(train_input).T, layer2_Delta)


if  __name__ == '__main__':
    neuralNet = neuralNetwork()
    print('Random starting weights :')
    print(neuralNet.weights)
    train_set = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    output_set = array([[0, 1 , 1, 0]]).T
    neuralNet.train_model(train_set, output_set, 10000)
    print('New weight after training: ')
    print(neuralNet.weight2)
    print('considering new input [0,0,0] -> ?: ')
    print(neuralNet.think(array([[1, 0, 0]])))
        
