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

            # reduced_input_to_layer = [tf.concat(input_to_layer[i * dilation:(i + 1) * dilation],axis=0) for i in range(reduced_timestamps)]
                
            reduced_output_from_layer, _ = tf.contrib.rnn.static_rnn(cell_layer, reduced_input_to_layer, dtype=tf.float32, scope = scope)
                
            output_from_layer = []
            for i in reduced_output_from_layer:
                tensor_split = tf.split(i, dilation,0)
                for j in range(len(tensor_split)):
                    output_from_layer.append(tensor_split[j])
                        
            input_to_layer = output_from_layer[:timestamps]
            layer += 1
        
        return input_to_layer
    

    def classification(self, input_data, class_num, timestamps, list_of_dilations, num_hidden_units, input_dimension, typeof_cell, experiment):
        
        if typeof_cell not in ["VanillaRNN", "LSTM", "GRU"]:
            raise ValueError("Valid cell type is 'VanillaRNN' or 'LSTM' or 'GRU'")
            
        cell_list = []
            
        if typeof_cell == "VanillaRNN":
            for i in range(len(list_of_dilations)):
                cell_tf = tf.contrib.rnn.BasicRNNCell(num_hidden_units)
                cell_list.append(cell_tf)
        elif typeof_cell == "LSTM":
            for i in range(len(list_of_dilations)):
                cell_tf = tf.contrib.rnn.BasicLSTMCell(num_hidden_units)
                cell_list.append(cell_tf)
        else:
            for i in range(len(list_of_dilations)):
                cell_tf = tf.contrib.rnn.GRUCell(num_hidden_units)
                cell_list.append(cell_tf)
        
        rnn_data = tf.unstack(input_data, timestamps, 1)
        
        outputs = self.drnn(cell_list, rnn_data, list_of_dilations)

        out_weights = tf.Variable(tf.random_normal(shape=[num_hidden_units, class_num]))
        out_bias = tf.Variable(tf.random_normal(shape=[class_num]))
        if experiment == "mnist":
            log_predictions = tf.add(tf.matmul(outputs[-1], out_weights), out_bias)
        elif experiment == "copy_memory":
            outputs = tf.stack(outputs, axis = 0)
            #shape = outputs.get_shape()
            #outputs = tf.reshape(outputs, [int(shape[1]), int(shape[0]), int(shape[2])])
            out_h = tf.einsum('ijk,kl->jil', outputs, out_weights)
            log_predictions = tf.add(out_h, out_bias)
        else:
            print("Wrong selection for the variable 'experiment'")
            
        return log_predictions

