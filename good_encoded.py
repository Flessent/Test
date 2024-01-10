import networkx as nx
import json
from pysat.formula import CNF
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
from sympy import symbols
from sympy.logic.boolalg import to_cnf
from math import ceil



class LinearLayer(layers.Layer):
    def __init__(self, num_neurons):
        super().__init__()
        self.num_neurons = num_neurons

    def build(self, input_shape):
        self.dense = layers.Dense(self.num_neurons)
     
    def call(self, inputs):
        print('LIN LAYER is called')
        return self.dense(inputs)


class BatchNormLayer(layers.Layer):
    def __init__(self, num_neurons, epsilon=1e-8):
        super().__init__()
        self.num_neurons = num_neurons
        self.epsilon = epsilon

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.num_neurons),
                                 initializer='random_normal',
                                 trainable=True)
        self.bias = self.add_weight(shape=(self.num_neurons,),
                                    initializer='zeros',
                                    trainable=True)

    def call(self, inputs):
        print('Batch LAYER is called')
        mean, var = tf.nn.moments(inputs, axes=[0])
        std = tf.sqrt(var + self.epsilon)

        normalized_inputs = (inputs - mean) / (std + self.epsilon)

        output = tf.matmul(normalized_inputs, self.W) + self.bias

        return output

class BINLayer(layers.Layer):
    def __init__(self, num_neurons):
        super().__init__()
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
        self.W = self.add_weight(shape=(input_shape[-1], self.num_neurons),
                                 initializer='random_normal',
                                 trainable=True)
        self.bias = self.add_weight(shape=(self.num_neurons,),
                                    initializer='zeros',
                                    trainable=True)

    def call(self, inputs):
        print('BIN Layer is called')
        binarized_output = self.binarize(inputs)

        #  STE to back propagate gradients through binarization
        straight_through_output = tf.stop_gradient(
            binarized_output - inputs) + inputs  # inputs argument here are real-valued weights

        return straight_through_output


class HardTanhLayer(tf.keras.layers.Layer):
    def __init__(self, num_neurons):
        super(HardTanhLayer, self).__init__()
        self.num_neurons=num_neurons

    def call(self, inputs):
        return tf.clip_by_value(inputs, -1, 1)


class ArgmaxLayer(layers.Layer):
    
    def __init__(self, num_neurons):
        super().__init__()
        self.num_neurons = num_neurons

    def call(self, inputs):
        print('ARGMAX LAYER is called')
        return tf.math.argmax(inputs, axis=-1)

class SoftmaxLayer(layers.Layer):
    def __init__(self, num_neurons):
        super().__init__()
        self.num_neurons = num_neurons

    def call(self, inputs):
        print('SOFTMAX LAYER is called')
        return tf.nn.softmax(inputs)
class InternalBlocks(models.Sequential):
    def __init__(self, num_blocks, num_neuron_lin=120, num_neuron_batch=120, num_neuron_bin=120, input_shape=(28,28), training=False):
        super().__init__()
        self.add(layers.Flatten(input_shape=input_shape))

        for _ in range(num_blocks):
            self.add(LinearLayer(num_neurons=num_neuron_lin))
            self.add(BatchNormLayer(num_neurons=num_neuron_batch))
            if training:
                self.add(HardTanhLayer(num_neurons=num_neuron_bin))
            self.add(BINLayer(num_neurons=num_neuron_bin))

class OutputBlocks(models.Sequential):
    def __init__(self, num_neuron_lin=10, num_neuron_argmax=10):
        super().__init__()
        self.add(LinearLayer(num_neurons=num_neuron_lin))
        self.add(SoftmaxLayer(num_neurons=num_neuron_argmax))


class BNN(tf.keras.Model):
    def __init__(self, num_internal_blocks, num_output_blocks, input_shape=(28, 28)):
        super().__init__()
        self.internal_blocks = InternalBlocks(num_blocks=num_internal_blocks, input_shape=input_shape)
        self.output_blocks = OutputBlocks()

    def call(self, inputs):

        x = self.internal_blocks(inputs)
        output = self.output_blocks(x)
        return output


def encode_bnn(bnn_model, cnf_formula):
    num_internal_blocks = len(bnn_model.internal_blocks.layers) // 3
    num_neurons_internal_block = 120
    num_neurons_output_block = 10

    var_to_int = {}
    count = 1

    def get_var_index(var):
        nonlocal count
        if var not in var_to_int:
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

    # Add boolean variables to the CNF formula
    for var in variables:
        cnf_formula.append([get_var_index(var), -get_var_index(var)])
        print(f"Added boolean variable: {var} or not {var}")

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
        counter_vars = [f"counter_{block_idx}_{neuron_idx}" for neuron_idx in range(num_neurons_internal_block)]
        sequential_counter_formula = [-get_var_index(counter_vars[0])]
        sequential_counter_formula += [get_var_index(counter_vars[j]) for j in range(num_neurons_internal_block)]
        cnf_formula.append(sequential_counter_formula)
        print(f"Added sequential counter clause: {sequential_counter_formula}")

        """"k_max = 5  # restricts the number of true literals to be at most K.
        if num_neurons_internal_block > k_max:
            at_most_k_formula = [-get_var_index(counter_vars[j]) for j in range(k_max + 1, num_neurons_internal_block)]
            cnf_formula.append(at_most_k_formula)
            print(f"Added At Most {k_max} constraint for internal block {block_idx}: {at_most_k_formula}")
            """



        for neuron_idx in range(num_neurons_internal_block):
            formula = binblk_formula(block_idx, neuron_idx) + [-get_var_index(counter_vars[neuron_idx])]
            cnf_formula.append(formula)
            print(f"Added clause: {formula}")

    counter_vars_output = [f"counter_output_{neuron_idx}" for neuron_idx in range(num_neurons_output_block)]
    sequential_counter_formula_output = [-get_var_index(counter_vars_output[0])]
    sequential_counter_formula_output += [get_var_index(counter_vars_output[j]) for j in range(num_neurons_output_block)]
    cnf_formula.append(sequential_counter_formula_output)
    print(f"Added sequential counter clause for output: {sequential_counter_formula_output}")

    for neuron_idx in range(num_neurons_output_block):
        formula = bino_formula(neuron_idx) + [-get_var_index(counter_vars_output[neuron_idx])]
        cnf_formula.append(formula)
        print(f"Added clause: {formula}")

    #print('CNF Formula Content', cnf_formula)
    return cnf_formula


def parse_bdd_file(file_path):
    bdd_info = {}

    with open(file_path, 'r') as file:
        lines = file.readlines()

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
                # Store all lines under the '.nodes' section
               bdd_info['nodes'] = lines[lines.index(line) + 1:]
                

    return bdd_info
def describe_network(model):
    for layer in model.internal_blocks.layers:
        if isinstance(layer, LinearLayer):
            print(f'for LinearLayer weights are : {layer.get_weights()}')
        elif isinstance(layer, BatchNormLayer):
            print(f'for BatchNormLayer weights are : {layer.get_weights()}')
        elif isinstance(layer, BINLayer):
            print(f'for BINLayer weights are : {layer.get_weights()}')
        elif isinstance(layer, layers.Flatten):
            print(f'Flatten layer, no weights to display')
        else:
            print(f'Unknown layer type: {type(layer)}')


def encode_bdd(bdd_structure, solver):
    num_vars = bdd_structure['nvars']
    num_nodes = bdd_structure['nnodes']

    var_to_int = {}
    count = 1

    def get_var_index(var):
        nonlocal count
        if var not in var_to_int:
            var_to_int[var] = count
            count += 1
        return var_to_int[var]

    variables = [f"node_{node}" for node in range(1, num_nodes + 1)]

    for var in variables:
        solver.add_clause([get_var_index(var), -get_var_index(var)])

    def bdd_node_formula(node):
        if bdd_structure['nodes'][node]['type'] == 'T':
            return [get_var_index(f"node_{node}")]
        else:
            var = f"node_{node}"
            low_child = bdd_structure['nodes'][node]['low']
            high_child = bdd_structure['nodes'][node]['high']
            return [get_var_index(var), -get_var_index(f"node_{low_child}"), get_var_index(f"node_{high_child}")]

    for node in range(1, num_nodes + 1):
        formula = bdd_node_formula(node)
        solver.add_clause(formula)

    return solver

def write_cnf_to_file(cnf_clauses, file_path='clauses.cnf'):
    with open(file_path, 'w') as file:
        for clause in cnf_clauses:
            file.write(" ".join(map(str, clause)) + " 0\n")
def read_cnf_from_file(file_path='clauses.cnf'):
    cnf_clauses = []
    with open(file_path, 'r') as file:
        for line in file:
            clause = list(map(int, line.split()[:-1])) 
            cnf_clauses.append(clause)
    return cnf_clauses

def train_bnn(model, train_data, train_labels, epochs=10, batch_size=64, validation_data=None):
   
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    if validation_data:
        history = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size,
                            validation_data=validation_data)
    else:
        history = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size)

    return history
def plot_the_network(the_model, resplt=10):
    the_weights = {}
    G = nx.Graph()
    the_colors = []
    previous_layer = []

    # Input layer (considering Flatten as a single node)
    G.add_node("input", layer=0)
    previous_layer += ["input"]
    the_colors += ["b"]

    # Internal blocks
    for block_idx, block in enumerate(the_model.internal_blocks.layers):
        new_layer = []

        # Flatten layer
        if isinstance(block.layers[0], tf.keras.layers.Flatten):
            node_name = f"{block_idx + 1}_FLATTEN"
            G.add_node(node_name, layer=block_idx + 1)
            new_layer += [node_name]
            the_colors += ["g"]

        # Linear layer within the block
        for layer in block.layers:
            if not isinstance(layer, tf.keras.layers.Flatten):
                for node in range(layer.num_neurons):
                    node_name = f"{block_idx + 1}_{layer.name}_{node + 1}"
                    G.add_node(node_name, layer=block_idx + 1)
                    new_layer += [node_name]
                    the_colors += ["r"]

        # Connect layers
        if new_layer:
            index2 = 0
            for node_o in previous_layer:
                index1 = 0
                for node_d in new_layer:
                    G.add_edge(node_o, node_d)
                    if "FLATTEN" not in node_o:
                        the_weights[(node_o, node_d)] = round(block.layers[1].get_weights()[0][index2][index1], 2)
                    index1 += 1
                index2 += 1

        previous_layer = new_layer

    # Output block
    new_layer = []
    for node in range(the_model.output_blocks.layers[1].num_neurons):
        node_name = f"output_{the_model.output_blocks.layers[1].name}_{node + 1}"
        G.add_node(node_name, layer=len(the_model.internal_blocks.layers) + 1)
        new_layer += [node_name]
        the_colors += ["r"]

    # Argmax layer in the output block
    node_name = "output_ARGMAX"
    G.add_node(node_name, layer=len(the_model.internal_blocks.layers) + 2)
    new_layer += [node_name]
    the_colors += ["r"]

    # Connect layers
    index2 = 0
    for node_o in previous_layer:
        index1 = 0
        for node_d in new_layer:
            G.add_edge(node_o, node_d)
            the_weights[(node_o, node_d)] = round(the_model.output_blocks.layers[1].get_weights()[0][index2][index1], 2)
            index1 += 1
        index2 += 1

    pos = nx.multipartite_layout(G, subset_key="layer")
    plt.clf()
    plt.figure(figsize=[resplt, resplt])

    # Draw nodes and edges
    nx.draw(G, pos, edge_color='black', width=2, linewidths=1,
            node_size=100 * resplt // (len(the_model.get_weights()) // 2),
            node_color=the_colors, alpha=0.9, with_labels=False)

    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=the_weights, label_pos=0.2, rotate=False,
                                 font_size=15 * resplt // ((len(the_model.get_weights()) // 2) + resplt))

    # Draw nodes again to have a cleaner plot
    nx.draw(G, pos, edge_color='black', width=2, edgelist=[], linewidths=1,
            node_size=100 * resplt // (len(the_model.get_weights()) // 2),
            node_color=the_colors, alpha=0.9, with_labels=False)

    plt.show()

if __name__ == "__main__":
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_train = X_train / 255.0
    X_test = X_test / 255.0
    process = psutil.Process(os.getpid())
    total_time = 0

    start_time = time.time()

    bnn_model = BNN(num_internal_blocks=1, num_output_blocks=1)
    #plot_the_network(bnn_model)
    print("NETWORK DESCRIPTION BEFORE TRAINING")
    describe_network(bnn_model)
    cnf_obj=CNF()
  
    cnf_formula = encode_bnn(bnn_model,cnf_obj)
    #encode_network(bnn_model)

    history = train_bnn(bnn_model, X_train, Y_train, epochs=50, batch_size=64, validation_data=(X_test, Y_test))
    print("NETWORK DESCRIPTION AFTER  TRAINING")
    describe_network(bnn_model)
    end_time = time.time()
    total_time = end_time - start_time

    training_loss = history.history['loss']
    training_accuracy = history.history['accuracy']
    epochs = range(1, len(training_loss) + 1)

    test_loss, test_acc = bnn_model.evaluate(X_test, Y_test)
    print('Test accuracy:', test_acc)

    memory_usage = process.memory_info().rss / 1024 / 1024  # in MB

    print(f'Total time taken for training: {total_time:.2f} seconds')
    print(f'Memory usage: {memory_usage:.2f} MB')

    
    predictions = bnn_model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    print('REAL VALUES ', Y_test[200])
    print('TEST PREDICTIONS',predicted_labels[200])
    accuracy = accuracy_score(Y_test, predicted_labels)
    f1 = f1_score(Y_test, predicted_labels, average='weighted')
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    plt.plot(epochs, training_loss, label='Training Loss')
    plt.plot(epochs, training_accuracy, label='Training Accuracy')

    plt.xlabel('Epochs')
    plt.ylabel('Loss / Accuracy')
    plt.title('Training Loss and Accuracy Over Epochs')
    plt.legend()
    plt.show()

    Y_pred_classes = np.argmax(predictions, axis=1)
    Y_true = Y_test
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
    f, ax = plt.subplots(figsize=(8, 8))
    sbn.heatmap(confusion_mtx, annot=True, linewidths=0.01, cmap="Greens", linecolor="gray", fmt='.1f', ax=ax)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    #plot_the_network(bnn_model)
   
    #write_cnf_to_file(cnf_formula, cnf_file)


    print("CNF Clauses:")
    #print(cnf_formula.clauses)
    write_cnf_to_file(cnf_formula.clauses)
    solver = Glucose3(bootstrap_with=read_cnf_from_file('generated_cnf_formula.cnf'))
    is_sat = solver.solve()
    if is_sat:
        print("BNN is satisfiable.")
        model_weights = [solver.get_model()[i] > 0 for i in range(len(solver.get_model()))]
        print("Model weights:", model_weights)
        print(f'CNF formula : {cnf_formula}')
        file_path='generated_cnf_formula.cnf'
        write_cnf_to_file(cnf_formula,file_path=file_path)

        print(f"CNF formula has been written to {file_path}")
    else:
       print("BNN is unsatisfiable.")