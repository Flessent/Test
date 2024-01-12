import tensorflow as tf
from tensorflow.keras import models,layers
from math import ceil
import numpy as np
import sys
from itertools import chain
import time





class LinearLayer(layers.Layer):
    def __init__(self, num_neurons):
        super().__init__()
        self.num_neurons = num_neurons

    def build(self, input_shape):
        self.dense = layers.Dense(self.num_neurons)
        

    def call(self, inputs):
        print('LIN layer is called')
        return self.dense(inputs)


class BatchNormLayer(layers.Layer):
    def __init__(self, num_neurons, epsilon=1e-8):
        super().__init__()
        self.num_neurons = num_neurons
        self.epsilon = epsilon

    def build(self, input_shape):
        # Initialize the weights and bias
        self.W = self.add_weight(shape=(input_shape[-1], self.num_neurons),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.bias = self.add_weight(shape=(self.num_neurons,),
                                    initializer='ones',
                                    trainable=True)

    def call(self, inputs):
        print('Batch Layer is called ')
        # Compute the mean and standard deviation
        mean, var = tf.nn.moments(inputs, axes=[0])
        std = tf.sqrt(var + self.epsilon)

        # Normalize the inputs
        normalized_inputs = (inputs - mean) / std

        # Compute the output
        output = tf.matmul(normalized_inputs, self.W) + self.bias

        return output



class BINLayer(layers.Layer):
    def __init__(self, num_neurons,**kwargs):
        super(BINLayer,self).__init__(**kwargs)
        self.num_neurons = num_neurons

    def binarize(self, inputs):
         #  saturated STE
        binarized_weights = tf.clip_by_value(self.W, -1.0, 1.0)
        binarized_bias = tf.clip_by_value(self.bias, -1.0, 1.0)

        binarized_weights = tf.where(self.W >= 0, 1.0, -1.0)
        binarized_bias = tf.where(self.bias >= 0, 1.0, -1.0)

        binarized_inputs = tf.where(inputs >= 0, 1.0, -1.0)

        # real-valued weights during backward pass (STE)
        
        #tf.print('Binarized Weights:', binarized_weights)
        #tf.print('Binarized Bias:', binarized_bias)
        #tf.print('Binarized Inputs:', binarized_inputs)
        binarized_output = tf.matmul(binarized_inputs, binarized_weights) + binarized_bias

        return binarized_output

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.num_neurons),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.bias = self.add_weight(shape=(self.num_neurons,),
                                    initializer='ones',
                                    trainable=True)
        super(BINLayer,self).build(input_shape)

    def call(self, inputs):
        print('BIN Layer is called')
        binary_weights = tf.sign(self.kernel)
        binary_inputs = tf.sign(inputs)
        binary_outputs = tf.matmul(binary_inputs, binary_weights)
        binary_outputs = tf.nn.bias_add(binary_outputs, self.bias)

        # Print binarized weights
        #tf.print("Binarized Weights:", binary_weights)

        return tf.nn.tanh(binary_outputs)




class ArgmaxLayer(layers.Layer):
    def __init__(self, num_neurons):
        super().__init__()
        self.num_neurons = num_neurons

    def call(self, inputs):
        return tf.math.argmax(inputs, axis=-1)


class HardTanhLayer(tf.keras.layers.Layer):
    def __init__(self, num_neurons):
        super(HardTanhLayer, self).__init__()
        self.num_neurons=num_neurons

    def call(self, inputs):
        return tf.clip_by_value(inputs, -1, 1)


class InternalBlocks(models.Sequential):
    def __init__(self, num_blocks, num_neuron_lin=50, num_neuron_batch=50, num_neuron_bin=50, input_shape=None, training=False):
        super().__init__()
        self.add(layers.InputLayer(input_shape=input_shape))

        for _ in range(num_blocks):
            self.add(LinearLayer(num_neurons=num_neuron_lin))
            self.add(BatchNormLayer(num_neurons=num_neuron_batch))
            if training:
                self.add(HardTanhLayer(num_neurons=num_neuron_bin))
            self.add(BINLayer(num_neurons=num_neuron_bin))

    def build(self, input_shape):
        super().build(input_shape)


class OutputBlocks(models.Sequential):
    def __init__(self, num_neuron_lin=50, num_neuron_argmax=2):
        super().__init__()
        self.add(LinearLayer(num_neurons=num_neuron_lin))
        self.add(ArgmaxLayer(num_neurons=num_neuron_argmax))


class BNN(tf.keras.Model):
    def __init__(self, num_internal_blocks, num_output_blocks, input_shape=(28,28)):
        super().__init__()
        self.internal_blocks = InternalBlocks(num_blocks=num_internal_blocks, input_shape=input_shape)
        self.output_blocks = OutputBlocks()

    def call(self, inputs):

        x = self.internal_blocks(inputs)
        output = self.output_blocks(x)
        return output

def seconds_separator(seconds_passed):
  duration = seconds_passed
  hours = int(duration//3600)
  duration = duration%3600
  minutes = int(duration//60)
  duration = float(round(duration%60, 4))
  return str(hours)+":"+"0"*(minutes<10)+str(minutes)+":"+"0"*(duration<10)+str(duration)+"0"*(6-len(str(duration))+(duration>=10))
np.set_printoptions(threshold=sys.maxsize)

def create_array(values:int=1, n_variables:int=1) -> np.array:
  return np.array([0]*(abs(values)-1)+[(values>0)*2-1]+[0]*(n_variables-abs(values)), dtype=np.int8).reshape(1,-1)

def delete_zeros(matrix:np.array, n_variables:int) -> np.array:
  return matrix[np.where((matrix == 0).sum(axis=1) != n_variables)[0]]

def simplify_cnf_formula(matrix:np.array, n_variables:int) -> np.array:
  simplified = matrix.copy()
  for x in range(n_variables-1, 0, -1):
    with_x_zeros = np.where((simplified == 0).sum(axis=1) == x)[0]
    if (with_x_zeros.size != 0) and ((simplified == 0).sum(axis=1).min() < x):
      comparables = simplified[with_x_zeros]
      erasables = simplified[np.where((simplified == 0).sum(axis=1) < x)[0]]
      with_more_zeros = np.where((simplified == 0).sum(axis=1) > x)[0]
      if (with_more_zeros.size != 0):
        saveables = simplified[with_more_zeros]
        for clause in comparables: erasables = erasables[np.where((erasables*(clause!=0) != clause).sum(axis=1) > 0)[0]]
        simplified = np.vstack([saveables, comparables, erasables])
      else:
        for clause in comparables: erasables = erasables[np.where((erasables*(clause!=0) != clause).sum(axis=1) > 0)[0]]
        simplified = np.vstack([comparables, erasables])
  return simplified

def conjunction_cnfs(matrix1:np.array, matrix2:np.array, n_variables:int) -> np.array:
  return simplify_cnf_formula(np.unique(np.vstack([matrix1, matrix2]), axis=0), n_variables)

def disjunction_cnfs(matrix1:np.array, matrix2:np.array, n_variables:int) -> np.array:
  new_ones = []
  for clause in matrix1:
    new_ones += [delete_zeros((matrix2+clause-clause*(matrix2==clause))*((matrix2*clause < 0).sum(axis=1) == 0).reshape(-1,1), n_variables)]
  return simplify_cnf_formula(np.unique(np.vstack(new_ones), axis=0), n_variables)

def cnf_negation(matrix: np.array, n_variables: int) -> np.array:
    final = np.zeros((1, n_variables))
    clauses = -matrix
    while clauses.shape[0] > 0:
        new_ones = []
        for ind_clause2 in range(min(n_variables, clauses.shape[1])):
            if clauses[0, ind_clause2] != 0:
                temp = create_array(clauses[0, ind_clause2] * (ind_clause2 + 1), n_variables)
                new_ones += [delete_zeros((final + temp - temp * (final == temp)) *
                                          ((final * temp < 0).sum(axis=1) == 0).reshape(-1, 1), n_variables)]
        final = np.unique(np.vstack(new_ones), axis=0)
        clauses = clauses[1:]

    return simplify_cnf_formula(final, n_variables)


def encode_network(the_model, input_file="BNN_CNFf.cnf") -> str:
  beginning = time.monotonic()
  n_inputs =  the_model.internal_blocks.layers[1].input_shape[1]
  inputs = [create_array(i, n_inputs) for i in range(1, n_inputs+1)]
  n_layer = 1 #  counter for tracking the current layer being processed
  num_internal_blocks = len(the_model.internal_blocks.layers) // 3
  for block in range(num_internal_blocks):
    for  layer  in the_model.internal_blocks.layers:
      #print('LAyer inputs :',layer.input_shape)

      the_weights = layer.get_weights()
      outputs = []
      for id_neuron in range(layer.num_neurons):
        print(f'{seconds_separator(time.monotonic() - beginning)}   Layer: {n_layer}/{len(the_model.internal_blocks.layers)-1} | Neuron: {id_neuron+1}/{layer.num_neurons}')
        print('BIN LAYER WEIGHTS NEURONS : ',the_model.internal_blocks.layers[2].get_weights()[1][id_neuron] )
        #D = ceil((-the_weights[1][id_neuron] + sum(the_weights[0][:,id_neuron])) / 2) + sum([(1 - weight) / 2 for weight in the_weights[0][:,id_neuron]]) # the_weights[1][id_neuron]   refers to the bias 
        print('BIAS :', (-the_weights[1][id_neuron]))
        print('SUM OF WEIGHTS :',the_weights[0][:,id_neuron].sum())
        print('LAST TERM : ',(the_weights[0][:,id_neuron] == -1).sum())

        D = max(1, ceil((-the_weights[1][id_neuron] + the_weights[0][:,id_neuron].sum())/2) + (the_weights[0][:,id_neuron] == -1).sum())
        print('D Values:', D)
        previous = {}
        #print('INputs :',inputs)
        for id_input in range(len(inputs)):
          if (layer == the_model.internal_blocks.layers[-1]): print(f'{seconds_separator(time.monotonic() - beginning)}   Working with the first {id_input+1} inputs')# check if it is the current layer is the last layer
          actual = {}
          if (the_weights[0][id_input,id_neuron] == 1): x = inputs[id_input] #the_weights[0] returns the first element containing the weight matrices.
          else: x = cnf_negation(inputs[id_input], len(inputs))
          for d in range(D):
            #print(f"D: {D}, d: {d}")
            #print(f"Previous keys: {list(previous.keys())}")
            #print(f"Target key: {target_key}, Previous keys: {list(previous.keys())}")

            if (id_input < d): break
            if (len(inputs) < id_input+1+D-(d+1)): continue
            if (d == 0):
              if (id_input == 0): actual[d] = x
              else: actual[d] = disjunction_cnfs(x, previous[d], len(inputs))
            elif (id_input == d): actual[d] = conjunction_cnfs(x, previous[d-1], len(inputs))
            else:
              temp = conjunction_cnfs(x, previous[d-1], len(inputs))
              actual[d] = disjunction_cnfs(temp, previous[d], len(inputs))
          previous = actual
          #print(f'D: {D}, d: {d}')
        outputs += [previous[D-1].astype(dtype=np.int8)]
        #print('Outputs:',outputs)
      inputs = outputs
      n_layer += 1
    print(f'Total time taken: {seconds_separator(time.monotonic() - beginning)}')
    dimacs_cnf = inputs[-1]
    dimacs_cnf = str(dimacs_cnf*np.arange(1,dimacs_cnf.shape[1]+1)).replace(" 0", "").replace("]","").replace("[","").replace("\n", " 0\n")+" 0\n"
    while "  " in dimacs_cnf: dimacs_cnf = dimacs_cnf.replace("  ", " ")
    output_file = "output_final.cnf"
    with open(output_file, 'w') as f:
      f.write('p cnf %d %d\n' % (n_inputs, dimacs_cnf.count('\n')))
      f.write(dimacs_cnf)
    #The simplifier has very recently stopped working on Colab, so we disabled it until we fix it
  return output_file

if __name__ == "__main__":
     bnn_model = BNN(num_internal_blocks=1, num_output_blocks=1)
     encode_network(bnn_model, input_file="BNN_CNFf.cnf")


