########################################################################################
    
### 4.1 COPY MEMORT PROBLEM ###

import numpy as np
import tensorflow as tf
from models.dilated_rnn import DilatedRNN
import time
import pandas as pd

class Dataset_Copy_Memory:
    
    def __init__(self, num_samples, T):
        self.num_samples = num_samples
        self.sample_len = T + 20
        self.T = T

        self.X_train, self.Y_train = self.generate(int(num_samples * 1))
        self.X_valid, self.Y_valid = self.generate(int(num_samples * 0.3))
        self.X_test, self.Y_test = self.generate(int(num_samples * 0.3))

    def generate(self, num_examples):
        X = np.ones((num_examples, self.sample_len, 1)) * 8
        nines = np.ones((num_examples, 11, 1)) * 9
        data = np.random.randint(low = 0, high = 7, size = (num_examples, 10, 1))
        X[:, :10] = data
        X[:, -11:] = nines
        X = X.astype(int)
        Y = np.ones((num_examples, self.sample_len, 1)) * 8
        Y[:, -10:] = X[:, :10]
        Y = Y.astype(int)
        Y = (np.squeeze(np.eye(10)[Y])).astype(int)
        
        return X, Y

    def get_train_data(self):
        return self.X_train, self.Y_train, self.X_valid, self.Y_valid, self.X_test, self.Y_test
    
    def get_validation_data(self):
        return self.X_valid, self.Y_valid

    def get_test_data(self):
        return self.X_test, self.Y_test
    
    def get_batch_count(self, batch_size):
        return self.X_train.shape[0] // batch_size

    def get_sample_len(self):
    	return self.sample_len

    def get_batch(self, batch_idx, batch_size):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size

        X_batch = self.X_train[start_idx:end_idx, :, :]
        Y_batch = self.Y_train[start_idx:end_idx, :, :]

        return X_batch, Y_batch


###############################################################################
###############################################################################
        

tf.reset_default_graph()

# Define some hyperparameters
T = [500, 1000]
num_examples = 20000 
num_of_layers = 9
num_input = 1
number_of_classes = 10 # 0-9 digits
cell_type_list = ["VanillaRNN", "LSTM", "GRU"]
hidden_units = 10 # hidden layer num of features
dilations = [2**j for j in range(num_of_layers)]
batch_size = 128
l_rate = 0.001
number_of_epochs = 80
experiment = "copy_memory"
n_test = 4000
decay = 0.9


for t_len in T:
    
    # Create the dataset
    dataset = Dataset_Copy_Memory(num_examples, t_len)
    # Compute the number of batches
    number_of_batches = dataset.get_batch_count(batch_size)
    # Validation data
    val_x, val_y = dataset.get_validation_data()
    # Training data
    test_x, test_y = dataset.get_test_data()
    
    unrolled_dim = t_len + 20
    
    for cell_type in cell_type_list:
        
        print("Starting new optimization process (" + experiment + ")")
        print("Model: Dilated " + cell_type + " for sequence length T= " + '{:d}'.format(t_len))
        
        tf.reset_default_graph()
    
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

        # number of trainable params
        t_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print('Number of trainable params= ' + '{:d}'.format(t_params))

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        with tf.Session() as sess:
            
            sess.run(init)
            
            results_train_set = []
            results_val_set = []
            
            start = time.time()
    
            for epoch in range(1, number_of_epochs+1):
        
                count_loss = 0
        
                for i in range(number_of_batches):
            
                    # Training set
                    batch_x, batch_y = dataset.get_batch(i, batch_size)

                    # Run optimization
                    batch_loss, _ = sess.run([loss_func, train], feed_dict={X_data: batch_x, y_labels: batch_y})
                    
                    count_loss += batch_loss
            
                train_loss = count_loss/number_of_batches
        
                # Run for validation set
                val_loss= sess.run(loss_func, feed_dict={X_data: val_x, y_labels: val_y})
        
                # Print results every 10 epochs
                if epoch % 10 == 0 or epoch == 1:
                    print("Epoch " + str(epoch) + ", Training Loss= " + \
                          "{:.4f}".format(train_loss) + ", Validation Loss= " + \
                          "{:.4f}".format(val_loss))
        
                results_train_set.append((epoch, train_loss))
                results_val_set.append((val_loss))
            
            print("Training Finished!")
            
            end = time.time()
            
            training_time = end - start
            
            print("Training time for this model: ", training_time)
            
            # Save the variables to disk.
            save_path = saver.save(sess, "path_of_the_file/Dilated_" + cell_type + "_" + str(t_len) + ".ckpt")
            print("Model saved in path: %s" % save_path)
    
            # Calculate cross entropy loss for copy memory
            
            testing_loss = sess.run(loss_func, feed_dict={X_data: test_x, y_labels: test_y})
        
            print("Testing Cross-Entropy Loss=" + "{:.3f}".format(testing_loss))
    

        # Storing our results to a dataframe
        results_train_set = pd.DataFrame(results_train_set)
        results_val_set = pd.DataFrame(results_val_set)
        
        results = pd.concat([results_train_set, results_val_set], axis=1, join='outer', ignore_index=False)
        results.columns = ["Epochs", "Training Loss", "Validation Loss"]
        
        export_csv = results.to_csv (r"path_of_the_file\Dilated_" + cell_type + "_" + str(t_len) + ".csv", index = None, header=True)
        
################################################### End of script #######################################################