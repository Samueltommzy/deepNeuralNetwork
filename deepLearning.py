import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
 

#one_hot is useful for multi-class classification
DATASETS = input_data.read_data_sets("/tmp/data", one_hot=True)

#set number of nodes in each layer of the network
LAYER_1, LAYER_2, LAYER_3 = 500, 500, 500

#set number of classes and batch size for our dataset
NUMBER_OF_CLASSES, BATCH_SIZE = 10, 100
X = tf.placeholder('float',[None, 784])
Y = tf.placeholder('float')

def model(dataset):
    """
    Model for neural network
    """
    #define weights and bias in each layer
    layer1 = {'weights': tf.Variable(tf.random_normal([784, LAYER_1])),
              'biases' : tf.Variable(tf.random_normal([LAYER_1]))
              }
    
    layer2 = {'weights': tf.Variable(tf.random_normal([LAYER_1, LAYER_2])),
              'biases' : tf.Variable(tf.random_normal([LAYER_2]))
              }
    
    layer3 = {'weights': tf.Variable(tf.random_normal([LAYER_2, LAYER_3])),
              'biases' : tf.Variable(tf.random_normal([LAYER_3]))
              }

    output_layer = {'weights': tf.Variable(tf.random_normal([LAYER_3, NUMBER_OF_CLASSES])),
              'biases' : tf.Variable(tf.random_normal([NUMBER_OF_CLASSES]))
              }

    layer_1_Value = tf.add((tf.matmul(dataset, layer1['weights'])), layer1['biases'])
    layer_1_Value = tf.nn.relu(layer_1_Value)

    layer_2_Value = tf.add((tf.matmul (dataset, layer2['weights'])), layer2['biases'])
    layer_2_Value = tf.nn.relu(layer_2_Value)

    layer_3_Value = tf.add((tf.matmul (dataset, layer3['weights'])), layer3['biases'])
    layer_3_Value = tf.nn.relu(layer_3_Value)
    
    output_Value = tf.add((tf.matmul (dataset, output_layer['weights'])), output_layer['biases'])
    return output_Value

def training(data):
    """
    function to train model from dataset
    """
    epochs = 5
    prediction = model(data)
    cost = tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y)) 
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    with tf.Session as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            epoch_loss = 0
            for _ in range(int(DATASETS.train.num_of_examples/BATCH_SIZE)):
                epoch_x, epoch_y = DATASETS.train.next_batch(BATCH_SIZE)
                _, c = sess.run([optimizer, cost], feed_dict={X: epoch_x, Y: epoch_y}) 
                epoch_loss += c
            print('Epoch', epoch, 'completed out of ', epochs, 'loss: ', epoch_loss)
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({X: DATASETS.test.images, Y: DATASETS.test.labels}))

training(X)
