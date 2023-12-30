import tensorflow as tf
import seaborn as sbn
import boto
from kiwisolver import Solver
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
from z3 import And, Bool, Solver, sat
import tensorflow as tf
from z3 import Real, Solver, Sum, Sqrt, Implies, Not, Bool

class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, num_neurons):
        super().__init__()
        self.num_neurons = num_neurons
        self.encoding_initialized = False

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(self.num_neurons)

        # Initialize encoding variables outside the tf.function
        input_size, output_size = input_shape[-1], self.num_neurons
        self.inputs = [Real(f'x_{i}') for i in range(input_size)]
        self.weight_vars = [[Real(f'w_{i}_{j}') for j in range(output_size)] for i in range(input_size)]
        self.bias_vars = [Real(f'b_{j}') for j in range(output_size)]
        self.output_vars = [Real(f'y_{j}') for j in range(output_size)]

        # Constraints
        constraints = []
        for j in range(output_size):
            sum_expr = self.bias_vars[j] + Sum([self.inputs[i] * self.weight_vars[i][j] for i in range(input_size)])
            constraints.append(self.output_vars[j] == sum_expr)

        # Store the constraints for later use
        self.constraints = constraints

    def call(self, inputs):
        # Check if encoding needs to be initialized
        if not self.encoding_initialized:
            self.initialize_encoding_variables(inputs)
            self.encoding_initialized = True

        return self.dense(inputs)

    def initialize_encoding_variables(self, inputs):
        # No need to initialize inside tf.function
        pass

    def get_tf_tensors(self):
        return {
            'inputs': tf.constant([float(x) for x in self.inputs]),
            'weight_vars': tf.constant([[float(x) for x in row] for row in self.weight_vars]),
            'bias_vars': tf.constant([float(x) for x in self.bias_vars]),
            'output_vars': tf.constant([float(x) for x in self.output_vars]),
        }

class BatchNormLayer(tf.keras.layers.Layer):
    def __init__(self, num_neurons, epsilon=1e-8):
        super().__init__()
        self.num_neurons = num_neurons
        self.epsilon = epsilon

    def build(self, input_shape):
        # Initialize the weights and bias
        self.W = self.add_weight(shape=(input_shape[-1], self.num_neurons),
                                 initializer='random_normal',
                                 trainable=True)
        self.bias = self.add_weight(shape=(self.num_neurons,),
                                    initializer='zeros',
                                    trainable=True)

    def call(self, inputs):
        # Encode BatchNormLayer
        output = self.encode_batchnorm_layer(inputs)

        return output

    def encode_batchnorm_layer(self, inputs):
        num_neurons = self.num_neurons
        input_size = int(inputs.shape[-1])

        # Variables
        inputs_z3 = [Real(f'x_{i}') for i in range(input_size)]
        mean, std = Real('mean'), Real('std')
        weights = [[Real(f'w_{i}_{j}') for j in range(num_neurons)] for i in range(input_size)]
        bias = [Real(f'b_{j}') for j in range(num_neurons)]
        output = [Real(f'y_{j}') for j in range(num_neurons)]

        # Constraints
        constraints = []
        constraints.append(mean == Sum(inputs_z3) / input_size)
        constraints.append(std == Sqrt(Sum([(inputs_z3[i] - mean) ** 2 for i in range(input_size)]) / input_size + self.epsilon))
        for j in range(num_neurons):
            constraints.append(output[j] == bias[j] + Sum([(inputs_z3[i] - mean) / std * weights[i][j] for i in range(input_size)]))

        # Debugging: Print the constraints
        print("Constraints:")
        for constraint in constraints:
            print(constraint)

        # Z3 solver
        solver = Solver()
        solver.add(constraints)

        # Debugging: Print solver state
        print("Solver state before check:", solver)
        
        # Check if the constraints are satisfiable
        if solver.check() != sat:
            raise ValueError("Constraints are unsatisfiable")

        # Debugging: Print solver state after check
        print("Solver state after check:", solver)

        # Extract Z3 variable values
        inputs_values = [solver.model().eval(var).as_decimal(10) for var in inputs_z3]
        weights_values = [[solver.model().eval(var).as_decimal(10) for var in weight_row] for weight_row in weights]
        bias_values = [solver.model().eval(var).as_decimal(10) for var in bias]
        output_values = [solver.model().eval(var).as_decimal(10) for var in output]

        # Convert values to TensorFlow tensors
        inputs_tf = tf.constant(inputs_values, dtype=tf.float32)
        weights_tf = tf.constant(weights_values, dtype=tf.float32)
        bias_tf = tf.constant(bias_values, dtype=tf.float32)
        output_tf = tf.constant(output_values, dtype=tf.float32)

        return inputs_tf, weights_tf, bias_tf, output_tf

class BINLayer(tf.keras.layers.Layer):
    def __init__(self, num_neurons):
        super().__init__()
        self.num_neurons = num_neurons

    def hard_tanh(self, x):
        return tf.clip_by_value(x, -1.0, 1.0)

    def binarize(self, inputs):
        # Encode BINLayer
        inputs, weights, bias, bin_inputs, bin_weights, bin_bias, bin_output, straight_through_output, constraints = self.encode_bin_layer(self)

        # Create Z3 solver
        solver = Solver()

        # Add constraints to the solver
        for constraint in constraints:
            solver.add(constraint)

        # Solve the system of constraints
        solver.check()
        model = solver.model()

        # Get the values of the variables
        inputs_values = [model.eval(var).as_decimal(10) for var in inputs]
        weights_values = [[model.eval(var).as_decimal(10) for var in weight_row] for weight_row in weights]
        bias_values = [model.eval(var).as_decimal(10) for var in bias]
        bin_inputs_values = [model.eval(var).as_decimal(10) for var in bin_inputs]
        bin_weights_values = [[model.eval(var).as_decimal(10) for var in bin_weight_row] for bin_weight_row in bin_weights]
        bin_bias_values = [model.eval(var).as_decimal(10) for var in bin_bias]
        bin_output_values = [model.eval(var).as_decimal(10) for var in bin_output]
        straight_through_output_values = [model.eval(var).as_decimal(10) for var in straight_through_output]

        return inputs_values, weights_values, bias_values, bin_inputs_values, bin_weights_values, bin_bias_values, bin_output_values, straight_through_output_values

    def encode_bin_layer(self, layer):
        num_neurons = layer.num_neurons
        input_size = layer.W.size()[0]

        # Variables
        inputs = [Real(f'x_{i}') for i in range(input_size)]
        weights = [[Real(f'w_{i}_{j}') for j in range(num_neurons)] for i in range(input_size)]
        bias = [Real(f'b_{j}') for j in range(num_neurons)]
        bin_inputs = [Real(f'bin_x_{i}') for i in range(input_size)]
        bin_weights = [[Real(f'bin_w_{i}_{j}') for j in range(num_neurons)] for i in range(input_size)]
        bin_bias = [Real(f'bin_b_{j}') for j in range(num_neurons)]
        bin_output = [Real(f'bin_y_{j}') for j in range(num_neurons)]
        straight_through_output = [Real(f'sto_y_{j}') for j in range(num_neurons)]

        # Constraints
        constraints = []
        for i in range(input_size):
            constraints.append(bin_inputs[i] == If(inputs[i] >= -1.0 and inputs[i] <= 1.0, inputs[i], If(inputs[i] < -1.0, -1.0, 1.0)))
            for j in range(num_neurons):
                constraints.append(bin_weights[i][j] == If(layer.W[i][j] >= 0, True, False))
        for j in range(num_neurons):
            constraints.append(bin_bias[j] == If(layer.bias[j] >= 0, True, False))
            constraints.append(bin_output[j] == bin_bias[j] + Sum([bin_inputs[i] * bin_weights[i][j] for i in range(input_size)]))
            constraints.append(straight_through_output[j] == bin_output[j] - inputs[j] + inputs[j])

        return inputs, weights, bias, bin_inputs, bin_weights, bin_bias, bin_output, straight_through_output, constraints

class ArgmaxLayer(tf.keras.layers.Layer):
    def __init__(self, num_neurons):
        super().__init__()
        self.num_neurons = num_neurons

    def call(self, inputs):
        # Encode ArgmaxLayer
        inputs, output, constraints = self.encode_argmax_layer(self)

        # Create Z3 solver
        solver = Solver()

        # Add constraints to the solver
        for constraint in constraints:
            solver.add(constraint)

        # Solve the system of constraints
        solver.check()
        model = solver.model()

        # Get the values of the variables
        inputs_values = [model.eval(var).as_decimal(10) for var in inputs]
        output_values = [model.eval(var).as_decimal(10) for var in output]

        return inputs_values, output_values

    def encode_argmax_layer(self, layer):
        num_neurons = layer.num_neurons

        # Variables
        inputs = [Real(f'input_{i}') for i in range(num_neurons)]
        output = Real('output')

        # Constraints
        constraints = []

        # Constraint: Argmax calculation
        for i in range(num_neurons):
            constraints.append(Implies(inputs[i] == Max(inputs), output == i))

        return inputs, output, constraints

# Build the model for training
def build_train_model(num_blocks):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    for _ in range(num_blocks - 1):  # Add 3 internal blocks
        model.add(LinearLayer(num_neurons=128))
        model.add(BatchNormLayer(num_neurons=128))
        model.add(BINLayer(num_neurons=128))
    model.add(LinearLayer(num_neurons=10))  # Add the output block
    model.add(ArgmaxLayer(num_neurons=10))
    return model

# Build the model for predictions
def build_predict_model(num_blocks):
    model = build_train_model(num_blocks)
    model.add(ArgmaxLayer(10))  # Add the ARGMAX layer
    return model

def main():
    # Load the data
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize the data
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    process = psutil.Process(os.getpid())
    total_time = 0

    start_time = time.time()

    num_blocks = 3  # Set the number of blocks
    # Build the model for training
    train_model = build_train_model(num_blocks)

    # Compile the model
    train_model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                        metrics=['accuracy'])

    # Train the model
    history = train_model.fit(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test), batch_size=128)

    end_time = time.time()
    total_time = end_time - start_time

    training_loss = history.history['loss']
    training_accuracy = history.history['accuracy']
    epochs = range(1, len(training_loss) + 1)

    # Evaluate the model
    test_loss, test_acc = train_model.evaluate(X_test, Y_test)
    print('Test accuracy:', test_acc)

    # Memory usage
    memory_usage = process.memory_info().rss / 1024 / 1024  # in MB

    print(f'Total time taken for training: {total_time:.2f} seconds')
    print(f'Memory usage: {memory_usage:.2f} MB')

    # Build the model for predictions
    predict_model = build_predict_model(num_blocks)

    # Copy the weights from the training model to the prediction model
    predict_model.set_weights(train_model.get_weights())

    # Prediction
    predictions = predict_model.predict(X_test)

    # Check non-deterministicity
    predictions2 = predict_model.predict(X_test)
    print("Non-deterministicity check:", np.all(predictions == predictions2))

    # Check robustness by adding noise to the input and seeing if the output changes significantly
    noise = np.random.normal(0, 0.01, X_test.shape)
    noisy_predictions = predict_model.predict(X_test + noise)
    print("Robustness check:", np.all(predictions == noisy_predictions))

    # Convert predictions to labels
    predicted_labels = np.argmax(predictions, axis=1)

    # Calculate accuracy and F1 score
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
    # Convert validation observations to one hot vectors
    Y_true = Y_test
    # compute the confusion matrix
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

    # plot the confusion matrix
    f, ax = plt.subplots(figsize=(8, 8))
    sbn.heatmap(confusion_mtx, annot=True, linewidths=0.01, cmap="Greens", linecolor="gray", fmt='.1f', ax=ax)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

if __name__ == "__main__":
    main()
   
