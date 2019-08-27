###########################################################################################
### 4.2 Pixel-by-pixel MNIST ###

import numpy as np
import tensorflow as tf
from models.dilated_rnn import DilatedRNN
import time
import pandas as pd

# Import mnist dataset from tensorflow
from tensorflow.examples.tutorials.mnist import input_data

mnist_dataset = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_data = mnist_dataset.train

# help(tf.contrib.learn.datasets.mnist.DataSet.next_batch)

tf.reset_default_graph()

# Define some hyperparameters
unrolled_dim = 784 # MNIST data input (img shape: 28*28) # timesteps
num_of_layers = 9
num_input = 1
number_of_classes = 10 # MNIST total classes (0-9 digits)
cell_type_list = ["VanillaRNN", "LSTM", "GRU"]
hidden_units = 20 # hidden layer num of features
dilations = [2**j for j in range(num_of_layers)]
batch_size = 128
l_rate = 0.001
number_of_epochs = 80
experiment = "mnist"
decay = 0.9
permutation_list = [True, False]

batch_number = train_data.num_examples//batch_size

for permutation in permutation_list:
    
    if permutation:
        np.random.seed(100)
        permute = np.random.permutation(784)
    
    # Validation set
    val_x = mnist_dataset.validation.images
    val_y = mnist_dataset.validation.labels
    if permutation:
        val_x = val_x[:, permute]
    val_x = val_x.reshape((-1, unrolled_dim, num_input))
    
    # Test set
    test_data = mnist_dataset.test.images
    test_label = mnist_dataset.test.labels
    if permutation:
        test_data = test_data[:, permute]
    test_data = test_data.reshape((-1, unrolled_dim, num_input))
    
    for cell_type in cell_type_list:
        
        if permutation:
            print("Starting new optimization process")
            print("Model: Dilated " + cell_type + " for permuted mnist")
        else:
            print("Starting new optimization process")
            print("Model: Dilated " + cell_type + " for unpermuted mnist")
            
            
        # Set the placeholders for our data
        X_data = tf.placeholder(tf.float32, [None, unrolled_dim, num_input])
        y_labels = tf.placeholder(tf.float32, [None, number_of_classes])

        # Retrieve the predictions
        pred_object = DilatedRNN()
        output_logits = pred_object.classification(X_data, number_of_classes, unrolled_dim, dilations, hidden_units, num_input, cell_type, experiment)

        # Loss function
        loss_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_logits, labels=y_labels))

        # Optimizer
        optimizer = tf.train.RMSPropOptimizer(l_rate, decay)
        train = optimizer.minimize(loss_func)

        # number of trainable params
        t_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print('Number of trainable params=', t_params)

        # Compute accuracy of the model
        probabilities = tf.nn.softmax(output_logits)
        predicted_class = tf.argmax(probabilities, 1)
        true_class = tf.argmax(y_labels, 1)
        equality = tf.equal(predicted_class, true_class)
        accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        with tf.Session() as sess:
    
            sess.run(init)
    
            results_train_set = []
            results_val_set = []
    
    
            start = time.time()
    
            for epoch in range(1,number_of_epochs+1):
        
                count_loss = 0
                count_accuracy = 0
        
                for _ in range(batch_number):
            
                    # Training set
                    batch_x, batch_y = train_data.next_batch(batch_size)
                    if permutation:
                        batch_x = batch_x[:, permute]
                    batch_x = batch_x.reshape((batch_size, unrolled_dim, num_input))

                    # Run optimization
                    batch_loss, batch_accuracy, _ = sess.run([loss_func, accuracy, train], feed_dict={X_data: batch_x, y_labels: batch_y})
        
                    count_loss += batch_loss
                    count_accuracy += batch_accuracy
            
                train_loss = count_loss/batch_number
                train_accuracy = count_accuracy/batch_number
        
                # Run for validation set
                val_loss, val_accuracy= sess.run([loss_func, accuracy], feed_dict={X_data: val_x, y_labels: val_y})

                results_train_set.append((epoch, train_loss, train_accuracy))
                results_val_set.append((val_loss, val_accuracy))
        
                # Print results every 10 epochs
                if epoch % 10 == 0 or epoch == 1:
                    print("Epoch " + str(epoch) + ", Training Loss= " + \
                          "{:.4f}".format(train_loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(train_accuracy) + ", Validation Loss= " + \
                          "{:.4f}".format(val_loss) + ", Validation Accuracy= " + \
                          "{:.3f}".format(val_accuracy))

            print("Training Finished!")
            
            end = time.time()
    
            training_time = end - start

            print("Training time for this model: ", training_time)
            
            # Save the variables to disk.
            if permutation:
                save_path = saver.save(sess, "path_of_the_file/Dilated_" + cell_type + "_permuted.ckpt")
                print("Model saved in path: %s" % save_path)  
            else:
                save_path = saver.save(sess, "path_of_the_file/Dilated_" + cell_type + "_unpermuted.ckpt")
                print("Model saved in path: %s" % save_path) 

    
            testing_acc = sess.run(accuracy, feed_dict={X_data: test_data, y_labels: test_label})
        
            print("Testing Accuracy=" + "{:.3f}".format(testing_acc))

    
        # Store our results
        results_train_set = pd.DataFrame(results_train_set)
        results_val_set = pd.DataFrame(results_val_set)
        
        results = pd.concat([results_train_set, results_val_set], axis=1, join='outer', ignore_index=False)
        results.columns = ["Epochs", "Training Loss", "Training Accuracy", "Validation Loss", "Validation Accuracy"]

        if permutation:
            export_csv = results.to_csv (r"path_of_the_file\Dilated_" + cell_type + "_permuted.csv", index = None, header=True)
        else:
            export_csv = results.to_csv (r"path_of_the_file\Dilated_" + cell_type + "_unpermuted.csv", index = None, header=True)
            
            
################################################### End of script #######################################################