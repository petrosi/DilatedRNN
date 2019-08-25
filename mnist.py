###############################################################################
### 4.2 Pixel-by-pixel MNIST ###

import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import pandas as pd
from models.dilated_rnn import DilatedRNN

# Import mnist dataset from tensorflow
from tensorflow.examples.tutorials.mnist import input_data

mnist_dataset = input_data.read_data_sets("MNIST_data/", one_hot=True)

tf.reset_default_graph()

# Define some hyperparameters
unrolled_dim = 784 # MNIST data input (img shape: 28*28) # timesteps
num_of_layers = 9
num_input = 1
number_of_classes = 10 # MNIST total classes (0-9 digits)
cell_type = "VanillaRNN"
hidden_units = 30 # hidden layer num of features
dilations = [2**j for j in range(num_of_layers)]
batch_size = 128
l_rate = 0.001
number_of_epochs= 4
experiment = "mnist"

### Unpermuted and permuted version of pixel-by-pixel mnist -- None because we have set the batch
# Set the placeholders for our data
X_data = tf.placeholder(tf.float32, [None, unrolled_dim, num_input])
y_labels = tf.placeholder(tf.float32, [None, number_of_classes])

# Retrieve the predictions
pred_object = DilatedRNN()
output_logits = pred_object.classification(X_data, number_of_classes, unrolled_dim, dilations, hidden_units, num_input, cell_type, experiment)

# Loss function
loss_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_logits, labels=y_labels))

# Optimizer
optimizer = tf.train.RMSPropOptimizer(l_rate, 0.9)
train = optimizer.minimize(loss_func)

# Compute accuracy of the model
probabilities = tf.nn.softmax(output_logits)
predicted_class = tf.argmax(probabilities, 1)
true_class = tf.argmax(y_labels, 1)
equality = tf.equal(predicted_class, true_class)
accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Optional
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.85)

permutation = True # if you want permuted version set it to true

if permutation:
    np.random.seed(100)
    permute = np.random.permutation(784)

# with tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)) as sess:

with tf.Session() as sess:
    
    sess.run(init)
    
    results_train_set = []
    results_val_set = []
    
    for epoch in range(1,number_of_epochs+1):
        # Training set
        batch_x, batch_y = mnist_dataset.train.next_batch(batch_size)
        if permutation:
            batch_x = batch_x[:, permute]
        batch_x = batch_x.reshape((batch_size, unrolled_dim, num_input))

        # Run optimization
        train_loss, train_accuracy, _ = sess.run([loss_func, accuracy, train], feed_dict={X_data: batch_x, y_labels: batch_y})
        
        # Validation set
        batch_x = mnist_dataset.validation.images
        batch_y = mnist_dataset.validation.labels
        if permutation:
            batch_x = batch_x[:, permute]
        batch_x = batch_x.reshape((-1, unrolled_dim, num_input))
        
        
        # Run optimization
        val_loss, val_accuracy= sess.run([loss_func, accuracy], feed_dict={X_data: batch_x, y_labels: batch_y})

        results_train_set.append((epoch, train_loss, train_accuracy))
        results_val_set.append((val_loss, val_accuracy))
        
        # Print results per 50 epochs
        
        if epoch % 2 == 0 or epoch == 1:
            print("Epoch " + str(epoch) + ", Training Loss= " + \
                  "{:.4f}".format(train_loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(train_accuracy) + ", Validation Loss= " + \
                  "{:.4f}".format(val_loss) + ", Validation Accuracy= " + \
                  "{:.3f}".format(val_accuracy))
    
    # Calculate accuracy for mnist test images
    test_data = mnist_dataset.test.images
    test_label = mnist_dataset.test.labels
    if permutation:
        test_data = test_data[:, permute]
    test_data = test_data.reshape((-1, unrolled_dim, num_input))
    
    testing_acc = sess.run(accuracy, feed_dict={X_data: test_data, y_labels: test_label})
        
    print("Testing Accuracy=" + "{:.3f}".format(testing_acc))
    
    print("Optimization Finished!")
    
    # Plotting our results
    results_train_set = pd.DataFrame(results_train_set)
    results_val_set = pd.DataFrame(results_val_set)
    
    results = pd.concat([results_train_set, results_val_set], axis=1, join='outer', ignore_index=False)
    results.columns = ["Epochs", "Training Loss", "Training Accuracy", "Validation Loss", "Validation Accuracy"]
    results['Epochs'] = pd.Categorical(results.Epochs)
    
    # Accuracy plot
    ax = plt.gca()

    results.plot(kind = 'line', x = 'Epochs', y = 'Training Accuracy', ax=ax)
    results.plot(kind = 'line', x = 'Epochs', y = 'Validation Accuracy', color='red', ax=ax)

    plt.show()
    
    # Validation plot
    ax = plt.gca()

    results.plot(kind = 'line', x = 'Epochs', y = 'Training Loss', ax=ax)
    results.plot(kind = 'line',x = 'Epochs', y = "Validation Loss", color='red', ax=ax)

    plt.show()

########################################################################################

