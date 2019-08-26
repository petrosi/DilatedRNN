import tensorflow as tf
import math

class DilatedRNN():
    
    def __init__(self):
        pass
    
    
    def drnn(self, tcell, x_data, dilation_list):
        
        input_to_layer = x_data.copy()
        
        layer = 1
        
        for dilation, cell_layer in zip(dilation_list, tcell):
            
            # Define the number of timestamps
            timestamps = len(input_to_layer)
        
            scope = "Layer_%d" %layer
            
            # We want dilation to devide exactly the timestamps
            if (timestamps % dilation == 0):
                # Reduce the sequence length by |dilation| times
                reduced_timestamps = timestamps//dilation
            else:
                reduced_timestamps = math.ceil(timestamps/dilation)
                added_timestamps = (reduced_timestamps * dilation) - timestamps
                zero_padding = tf.zeros_like(input_to_layer[0])
                for i in range(added_timestamps):
                    input_to_layer.append(zero_padding)
        
            reduced_input_to_layer = []

            for i in range(0,len(input_to_layer), dilation):
                concat_tensors = tf.concat(input_to_layer[i:(i+dilation)], 0)
                reduced_input_to_layer.append(concat_tensors)
                
            reduced_output_from_layer, _ = tf.contrib.rnn.static_rnn(cell_layer, reduced_input_to_layer, dtype=tf.float32, scope = scope)
                
            output_from_layer = []
            for i in reduced_output_from_layer:
                tensor_split = tf.split(i, dilation,0)
                for j in range(len(tensor_split)):
                    output_from_layer.append(tensor_split[j])
                        
            input_to_layer = output_from_layer[:timestamps]
            layer += 1
        
        return input_to_layer
    

    def classification(self, input_data, class_num, timestamps, list_of_dilations, num_hidden_units, input_dimension, typeof_cell, experiment, dropout = None):
        
        if typeof_cell not in ["VanillaRNN", "LSTM", "GRU"]:
            raise ValueError("Valid cell type is 'VanillaRNN' or 'LSTM' or 'GRU'")
        
        type_to_func_dict = {
                "VanillaRNN": tf.contrib.rnn.BasicRNNCell,
                "LSTM": tf.contrib.rnn.BasicLSTMCell,
                "GRU": tf.contrib.rnn.GRUCell
        }
        
        cell_func = type_to_func_dict[typeof_cell]
        cell_list = [cell_func(num_hidden_units) for _ in range(len(list_of_dilations))]
        
        if dropout is not None:
            cell_list[0] = tf.contrib.rnn.DropoutWrapper(cell_list[0], dropout, 1, 1)
        
        rnn_data = tf.unstack(input_data, timestamps, 1)
        
        outputs = self.drnn(cell_list, rnn_data, list_of_dilations)

        
        if experiment == "mnist":
            start_dilation = list_of_dilations[0]
            if start_dilation == 1:
                out_weights = tf.Variable(tf.random_normal(shape=[num_hidden_units, class_num]))
                out_bias = tf.Variable(tf.random_normal(shape=[class_num]))
                fuse_outputs = outputs[-1]
            else:
                out_weights = tf.Variable(tf.random_normal(shape=[num_hidden_units*start_dilation, class_num]))
                out_bias = tf.Variable(tf.random_normal(shape=[class_num]))
                fuse_outputs = outputs[-start_dilation]
                for i in range(-start_dilation+1, 0, 1):
                    fuse_outputs = tf.concat([fuse_outputs, outputs[i]], axis = 1)
            
            log_predictions = tf.add(tf.matmul(fuse_outputs, out_weights), out_bias)
                
        elif experiment == "copy_memory" or experiment == "PTB":
            
            out_weights = tf.Variable(tf.random_normal(shape=[num_hidden_units, class_num]))
            out_bias = tf.Variable(tf.random_normal(shape=[class_num]))
            
            outputs = tf.stack(outputs, axis = 0)
            out_h = tf.einsum('ijk,kl->jil', outputs, out_weights)
            log_predictions = tf.add(out_h, out_bias)
            
        else:
            
            print("Wrong selection for the variable 'experiment'")
            
        return log_predictions
