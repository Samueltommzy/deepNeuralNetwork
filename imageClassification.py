import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag','Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0
# %matplotlib inline
# plt.figure(figsize = (10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid('off')
#     plt.imshow(test_images[i], cmap = plt.cm.get_cmap())
#     plt.xlabel(class_names[test_labels[i]])
# plt.show()


model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(128, activation = (tf.nn.relu)),
    keras.layers.Dense(20,activation=(tf.nn.softmax)),
    keras.layers.Dense(10,  activation = (tf.nn.softmax))
])

print( model.weights)
model.compile(optimizer = tf.train.AdamOptimizer(), loss = 'sparse_categorical_crossentropy', metrics=(['accuracy']))
model.fit(train_images,train_labels,epochs=8)

test_loss,test_accuracy = model.evaluate(test_images,test_labels)
print("test accuracy...... : ", test_accuracy)

predictions = model.predict(test_images)
print("predictions...", predictions)
print(np.argmax(predictions[18]))
print(test_labels[18])

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(test_images[i],cmap=plt.cm.get_cmap())
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]
    if predicted_label == true_label:
        color = 'green'
    else:
        color = "red"
    plt.xlabel("{} ({})".format(class_names[predicted_label],class_names[true_label]),color=color)
plt.show()