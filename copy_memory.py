########################################################################################
    
### 4.1 COPY MEMORT PROBLEM ###

import numpy as np
import tensorflow as tf
from models.dilated_rnn import DilatedRNN

# Function that generates the data for the copy problem
def sequence_gen(T, batch_num):
    
    X = np.ones((batch_num, T + 20, 1)) * 8
    nines = np.ones((batch_num, 11, 1)) * 9
    data = np.random.randint(low = 0, high = 7, size = (batch_num, 10, 1))
    X[:, :10] = data
    X[:, -11:] = nines
    X = X.astype(int)
    Y = np.ones((batch_num, T + 20, 1)) * 8
    Y[:, -10:] = X[:, :10]
    Y = Y.astype(int)
    Y = (np.squeeze(np.eye(10)[Y])).astype(int)
    
    return X, Y


tf.reset_default_graph()

# Define some hyperparameters
T = 100
unrolled_dim = T + 20 
num_of_layers = 9
num_input = 1
number_of_classes = 10 # 0-9 digits
cell_type = "VanillaRNN"
hidden_units = 20 # hidden layer num of features
dilations = [2**j for j in range(num_of_layers)]
batch_size = 128
l_rate = 0.001
number_of_epochs= 4
experiment = "copy_memory"
n_test = 1000
decay = 0.9


### Unpermuted and permuted version of pixel-by-pixel mnist -- None because we have set the batch
# Set the placeholders for our data
X_data = tf.placeholder(tf.float32, [None, unrolled_dim, num_input])
y_labels = tf.placeholder(tf.float32, [None, unrolled_dim, number_of_classes])

# Retrieve the predictions
pred_object = DilatedRNN()
output_logits = pred_object.classification(X_data, number_of_classes, unrolled_dim, dilations, hidden_units, num_input, cell_type, experiment)

# Loss function
loss_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_logits, labels=y_labels))

# Optimizer
optimizer = tf.train.RMSPropOptimizer(l_rate, decay)
train = optimizer.minimize(loss_func)


# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


with tf.Session() as sess:
    
    sess.run(init)
    
    results_train_set = []
    
    for epoch in range(1,number_of_epochs+1):
        
        # Training set
        batch_x, batch_y = sequence_gen(T, batch_size)

        # Run optimization
        train_loss, _ = sess.run([loss_func, train], feed_dict={X_data: batch_x, y_labels: batch_y})

        results_train_set.append((epoch, train_loss))
        
        # Print results per 50 epochs
        
        if epoch % 2 == 0 or epoch == 1:
            print("Epoch " + str(epoch) + ", Training Loss= " + \
                  "{:.4f}".format(train_loss))
    
    # Calculate cross entropy loss for copy memory
    test_x, test_y = sequence_gen(T, n_test)
    
    testing_loss = sess.run(loss_func, feed_dict={X_data: test_x, y_labels: test_y})
        
    print("Testing Cross-Entropy Loss=" + "{:.3f}".format(testing_loss))
    
    print("Optimization Finished!")


#############################################################################################
