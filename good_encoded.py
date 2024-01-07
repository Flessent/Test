from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import seaborn as sbn
import boto
#from kiwisolver import Solver
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from tensorflow.keras import models, layers
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import os
import psutil
import time
import sys
import ast
from pysat.solvers import Glucose3
from itertools import product
from pysat.solvers import Solver
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from tensorflow.keras.layers import Layer
from tensorflow.keras import activations

class LinearLayer(Layer):
    def __init__(self, num_neurons):
        super().__init__()
        self.num_neurons = num_neurons

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.num_neurons),
                                 initializer='random_normal',
                                 trainable=True)
        self.bias = self.add_weight(shape=(self.num_neurons,),
                                    initializer='zeros',
                                    trainable=True)

    def call(self, inputs):
        print('LIN LAYER is called')
        return tf.matmul(inputs, self.W) + self.bias

class BatchNormLayer(Layer):
    def __init__(self, num_neurons, epsilon=1e-8):
        super().__init__()
        self.num_neurons = num_neurons
        self.epsilon = epsilon

    def build(self, input_shape):
        self.alpha = self.add_weight(shape=(self.num_neurons,),
                                     initializer='ones',
                                     trainable=True)
        self.gamma = self.add_weight(shape=(self.num_neurons,),
                                     initializer='zeros',
                                     trainable=True)
        self.mu = self.add_weight(shape=(self.num_neurons,),
                                  initializer='zeros',
                                  trainable=False)
        self.sigma = self.add_weight(shape=(self.num_neurons,),
                                     initializer='ones',
                                     trainable=False)

    def call(self, inputs, training=None):
        print('Batch LAYER is called')

        if training:
            mean, var = tf.nn.moments(inputs, axes=[0])
            self.mu.assign(mean)
            self.sigma.assign(tf.sqrt(var + self.epsilon))
        
        normalized_inputs = (inputs - self.mu) / (self.sigma + self.epsilon)
        output = self.alpha * normalized_inputs + self.gamma

        return output

class BINLayer(Layer):
    def __init__(self, num_neurons):
        super().__init__()
        self.num_neurons = num_neurons

    def sSTE(self, x):
        return tf.clip_by_value(x, -1.0, 1.0)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.num_neurons),
                                 initializer='random_normal',
                                 trainable=True)
        self.bias = self.add_weight(shape=(self.num_neurons,),
                                    initializer='zeros',
                                    trainable=True)

    def call(self, inputs):
        print('BIN LAYER is called')
        binarized_weights = self.sSTE(self.W)
        binarized_bias = self.sSTE(self.bias)
        binarized_inputs = self.sSTE(inputs)

        binarized_output = tf.matmul(binarized_inputs, binarized_weights) + binarized_bias
        straight_through_output = tf.stop_gradient(binarized_output - inputs) + inputs

        return straight_through_output

class ArgmaxLayer(Layer):
    def __init__(self, num_neurons):
        super().__init__()
        self.num_neurons = num_neurons

    def call(self, inputs):
        print('ARGMAX LAYER is called')
        return tf.argmax(inputs, axis=-1)

class HardTanhLayer(Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return tf.clip_by_value(inputs, -1.0, 1.0)

class InternalBlocks(models.Sequential):
    def __init__(self, num_blocks, num_neuron_lin=120, num_neuron_batch=120, num_neuron_bin=120, add_hardtanh=False):
        super().__init__()
        self.add(layers.Flatten(input_shape=(28, 28)))

        for _ in range(num_blocks):
            self.add(LinearLayer(num_neurons=num_neuron_lin))
            self.add(BatchNormLayer(num_neurons=num_neuron_batch))
            if add_hardtanh:
                self.add(HardTanhLayer())
            self.add(BINLayer(num_neurons=num_neuron_bin))

            

class OutputBlocks(models.Sequential):
    def __init__(self, num_neuron_lin=10, num_neuron_argmax=10):
        super().__init__()
        self.add(LinearLayer(num_neurons=num_neuron_lin))
        self.add(ArgmaxLayer(num_neurons=num_neuron_argmax))

# Modify the BNN class to include the possibility of adding a HardTanhLayer
class BNN(tf.keras.Model):
    def __init__(self, num_internal_blocks, num_output_blocks, add_hardtanh=False):
        super().__init__()
        self.internal_blocks = InternalBlocks(num_blocks=num_internal_blocks, add_hardtanh=add_hardtanh)
        self.output_blocks = OutputBlocks()


    def call(self, inputs):
        x = self.internal_blocks(inputs)
        output = self.output_blocks(x)
        return output

# CNF encoding of  the BNN using PySAT
variables = []
def encode_bnn(bnn_model, solver):
    global variables
    num_internal_blocks = len(bnn_model.internal_blocks.layers) // 3  # Each internal block has 3 layers
    num_neurons_internal_block = 50  #  same number of neurons in each internal block
    num_neurons_output_block = 10  # same number of neurons in the output block
    print('In the encode function')

   
    var_to_int = {} # Map variable names to integers
    count = 1  # assigning unique integers to variables.

   

    def get_var_index(var):
        nonlocal count
        if var not in var_to_int:
            print('variables names',var)
            var_to_int[var] = count
            count += 1
        
        return var_to_int[var]

    # Boolean variables for neurons
    variables = [
        f"neuron_{block}_{layer}_{neuron}"
        for block, layer, neuron in product(range(num_internal_blocks), ["LIN", "BN", "BIN"], range(num_neurons_internal_block))
    ]
    
    variables += [
        f"output_neuron_{layer}_{neuron}"
        for layer, neuron in product(["LIN", "ARGMAX"], range(num_neurons_output_block))
    ]
      

    # Add boolean variables to the solver
    for var in variables:
        # Add a clause with both positive and negative literals
        solver.add_clause([get_var_index(var), -get_var_index(var)])

        #solver.add_clause([get_var_index(var)])
        #solver.add_clause([-get_var_index(var)])
        print(f"Added boolean variable: {var} or not {var}")

    # Boolean functions BINBLK and BINO
    def binblk_formula(block_idx, neuron_idx):
        lin_var = f"neuron_{block_idx}_LIN_{neuron_idx}"
        bn_var = f"neuron_{block_idx}_BN_{neuron_idx}"
        bin_var = f"neuron_{block_idx}_BIN_{neuron_idx}"
        return [get_var_index(lin_var), get_var_index(bn_var), get_var_index(bin_var)]

    def bino_formula(neuron_idx):
        lin_var = f"output_neuron_LIN_{neuron_idx}"
        argmax_var = f"output_neuron_ARGMAX_{neuron_idx}"
        return [get_var_index(lin_var), get_var_index(argmax_var)]

    # Add clauses corresponding to BINBLK and BINO
   
    for block_idx in range(num_internal_blocks):
        for neuron_idx in range(num_neurons_internal_block):
            formula = binblk_formula(block_idx, neuron_idx)
            solver.add_clause(formula)
            print(f"Added clause: {formula}")

    for neuron_idx in range(num_neurons_output_block):
        formula = bino_formula(neuron_idx)
        solver.add_clause(formula)
        print(f"Added clause: {formula}")
    print('Solver Content',solver)
    return solver

def parse_bdd_content(bdd_content):
    bdd_info = {}

    # Split the content into lines
    lines = bdd_content.split('\n')

    # Extract information from the lines
    for line in lines:
        if line.startswith('.nnodes'):
            bdd_info['nnodes'] = int(line.split()[1])
        elif line.startswith('.nvars'):
            bdd_info['nvars'] = int(line.split()[1])
        elif line.startswith('.nsuppvars'):
            bdd_info['nsuppvars'] = int(line.split()[1])
        elif line.startswith('.ids'):
            bdd_info['ids'] = list(map(int, line.split()[1:]))
        elif line.startswith('.permids'):
            bdd_info['permids'] = list(map(int, line.split()[1:]))
        elif line.startswith('.nroots'):
            bdd_info['nroots'] = int(line.split()[1])
        elif line.startswith('.rootids'):
            bdd_info['rootids'] = int(line.split()[1])
        elif line.startswith('.nodes'):
            # Stop parsing when '.nodes' is encountered
            break

    return bdd_info

def read_bdd_file(file_path):
    with open(file_path, 'r') as file:
        bdd_content = file.read()
    return bdd_content



def preprocess_data():
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Flatten images
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))

    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test

def train_bnn(model, x_train, y_train, solver, num_epochs=5):
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Iterate over the training dataset
        for i in range(len(x_train)):
            inputs = x_train[i]
            labels = y_train[i]

            # Encode BNN using PySAT
            solver = encode_bnn(model, solver)

            # Solve the encoded BNN
            is_sat = solver.solve()

            if is_sat:
                print("BNN is satisfiable.")
                model_values = solver.get_model()

                # Update weights using the model_values and labels
                update_weights(model, model_values, labels)

            else:
                print("BNN is unsatisfiable. Clauses added to the solver.")

    print("Training complete.")
def update_weights(model, model_values, labels, learning_rate=0.01):
    # Iterate over layers and update weights
    for layer in model.layers:
        if isinstance(layer, LinearLayer) or isinstance(layer, BatchNormLayer) or isinstance(layer, BINLayer):
            weights = layer.get_weights()
            
            # Extract values from the model_values and update weights
            if isinstance(layer, BatchNormLayer):
                layer.alpha.assign(model_values['alpha'])
                layer.gamma.assign(model_values['gamma'])
                layer.mu.assign(model_values['mu'])
                layer.sigma.assign(model_values['sigma'])
            else:
                layer.W.assign(model_values['W'])
                layer.bias.assign(model_values['bias'])

if __name__ == "__main__":
    # Load and preprocess data
    x_train, y_train, x_test, y_test = preprocess_data()

    # Create BNN model
    bnn_model = BNN(num_internal_blocks=1, num_output_blocks=1,add_hardtanh=False)

    # Create PySAT solver
    solver = Glucose3()

    # Train the BNN
    train_bnn(bnn_model, x_train, y_train, solver, num_epochs=5)

    # After training, you can use the trained model for predictions, evaluation, etc.
    # Example: Make predictions on test data
    predictions = bnn_model.predict(x_test)

    # Example: Evaluate the model
    accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1))
    print(f"Test Accuracy: {accuracy}")
