import numpy as np
#import lasagne
#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import tensorflow as tf
import random

class BatchNormalization():
      def __init__(self, X, gamma, beta, bn_param):
            self.X=X #input data for testing or for training
            self.gamma=gamma
            self.beta=beta
            self.bn_param=bn_param
      
      
      def batchnorm_forward(self):
            """
            Forward pass for batch normalization.

            Input:
            - x: Data of shape (N, D)
            - gamma: Scale parameter of shape (D,)
            - beta: Shift paremeter of shape (D,)
            - bn_param: Dictionary with the following keys:
            - mode: 'train' or 'test'; required
            - eps: Constant for numeric stability
            - momentum: Constant for running mean / variance.
            - running_mean: Array of shape (D,) giving running mean of features
            - running_var Array of shape (D,) giving running variance of features

            Returns a tuple of:
            - out: of shape (N, D)
            - cache: A tuple of values needed in the backward pass
            """
            
            mode = self.bn_param['mode']
            eps = self.bn_param.get('eps', 1e-5)
            momentum = self.bn_param.get('momentum', 0.9)
            print('X in feed forward',self.X)

            N, D = self.X.shape
            running_mean = self.bn_param.get('running_mean', np.zeros(D, dtype=self.X.dtype))
            running_var = self.bn_param.get('running_var', np.zeros(D, dtype=self.X.dtype))

            out, cache = None, None
            if mode == 'train':
                  sample_mean = np.mean(self.X, axis=0)
                  sample_var = np.var(self.X, axis=0)

                  # Normalization followed by Affine transformation
                  x_normalized = (self.X - sample_mean) / np.sqrt(sample_var + eps)
                  print('Normalized Batch in feed forward',x_normalized)
                  print('Gamma',self.gamma)
                  print('Beta',self.beta)
                  out = self.gamma * x_normalized + self.beta

                  # Estimate running average of mean and variance to use at test time
                  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
                  running_var = momentum * running_var + (1 - momentum) * sample_var

                  # Cache variables needed during backpropagation
                  cache = (self.X, sample_mean, sample_var, self.gamma, self.beta, eps)

            elif mode == 'test':
                  # normalize using running average
                  x_normalized = (self.X - running_mean) / np.sqrt(running_var + eps)

                  # Learned affine transformation
                  out = self.gamma * x_normalized + self.beta

                  # Store the updated running means back into bn_param for normalizing input data during testing
                  self.bn_param['running_mean'] = running_mean
                  self.bn_param['running_var'] = running_var

            return out, cache
class Binarization():
    def __init__(self,H=1,bias=1,threshold=0):
        self.H=H
        self.bias=bias
        self.threshold=threshold

    def signum(self,variable): #variable could be weigths or activation
            print('Type in signum', (variable))
            variable = np.where(variable >= self.threshold, self.H, -self.H)

            return variable
    def hard_sigmoid(self,x):
          return np.clip((x+1.)/2.,0,1)

    def Binarize_Deterministic(self,W,b):
            #self.threshold,self.H,lower=0,1,-1
            Wb=np.zeros(W.shape)
            bb=np.zeros(b.shape)
            Wb[W>=self.threshold]=self.H
            Wb[W<self.threshold]=-self.H
            bb[b>=self.threshold]=self.H
            bb[b<self.threshold]=-self.H
            return Wb,bb
    def hard_sigmoid(self,x):
          return np.clip((x+1.)/2.,0,1)

    def Binarize_Deterministic_A(self,Z): #Binarization function for activations using a deterministic approach.
            # Input: activation A 
            # Output: Binarized A
            #threshold,upper,lower=0,1,-1
            # A=Z= Raw score or logits=> Z =W*X +b 
            Zb=np.zeros(Z.shape)	
            Zb[Z>=self.threshold]=self.H
            Zb[Z<self.threshold]=-self.H
            return Zb  # binarized logits
    
def Binarize_Stochastic(self,W,b):	
      seed = 123
      rng = tf.random.Generator.from_seed(seed)
      random_int = rng.uniform((1,), minval=1, maxval=21, dtype=tf.int32)
      Wb = self.hard_sigmoid(W/1.0)
      print( Wb.shape)
      Wb[Wb>=0.5]=1
      Wb[Wb<=0.5]=-1	
      bb = self.hard_sigmoid(b/1.0)
      return Wb,bb


class Neuron():
      def __init__(self,bias):
            self.bias=bias
            self.weights=[] #np.zeros( (10,2) )
            self.inputs=[] #np.zeros( (2,10)  )
            self.outputs=[] #np.zeros( (10,2)  )
            
     
      def calculate_total_net_input(self):
        Z=0# Z=W*X +b
        print("Contents inputs :",self.inputs)
        print("contents weights :",self.weights)
        print("size inputs :",len(self.inputs))
        print('SIZE Weights  ', len(self.weights))
        for i in np.arange(len(self.inputs)):
                  
                  #print('Inputs content',self.inputs[i] )
                  #print('result:',(np.array( [item for sublist in self.inputs[i] for item in sublist])))
                  #Z+=self.weights[i] * np.array( [item for sublist in self.inputs[i] for item in sublist] )
                  #print('Shape Weights', self.weights[i])
                  #print('Shape Inputs', self.inputs[i])
                  Z += np.dot(self.weights[i], self.inputs[i]) +self.bias

        return Z
      


      """

      def activation(self,Z):# Z=total_net_input
            #Z=self.calculate_total_net_input()
            return 1/(1+np.exp(-Z))
            
      """
               
          
      def calculate_output(self, inputs):    
        binarization=Binarization(H=1,threshold=0,bias=self.bias)                              
        self.inputs = inputs
        print("Calculate Outputs :",self.inputs)
        self.outputs = binarization.signum(self.calculate_total_net_input())
        print('Binarized outputs',self.outputs)
        return self.outputs
      
      # Backpropagation using chain rule : d(total_error)/d(weight)=dE/(d(output_after_activation))*(d(output_after_activation)/d(dZ))*(dZ/d(weight))
      def calculate_pd_error_wrt_total_net_input(self, target_output):
            print('target output',target_output)
            print('pd_error',self.calculate_pd_error_wrt_output(target_output))
            print('wrt Z',self.calculate_der_output_neuron_wrt_Z())
            #print("content in cal_pd_err with reshape", (self.calculate_pd_error_wrt_output(target_output) * self.calculate_der_output_neuron_wrt_Z()).reshape(2,1).flatten().shape)
            return self.calculate_pd_error_wrt_output(target_output) * self.calculate_der_output_neuron_wrt_Z() 

      def calculate_error_of_each_neuron(self,target_output):#Using MSE
            return  0.5*(self.outputs-target_output)**2
      def calculate_pd_error_wrt_output(self,target_output): # partial derivative of total error w.r.t each ouputs of from each neuron : dE/d(output_after_activation)
            return (self.outputs-target_output)
             # During forward pass we got : ouput_neuron=1/1+ exp(-total_net_inputs)
            # During BckProg using chain rule we partially derive total error w.r.t. the output of this neuron
            #then the derivative of the output(after activation function) w.r.t the total nets inputs so-called Z=W*X +b
            # This derivative looks like : d(output_after_activation)/d(Z) 
            # we note output_after_activation : out_after_act 
      def calculate_der_output_neuron_wrt_Z(self):#d(output_after_activation)/d(dZ)
            return self.outputs*(1-self.outputs)
      
      def calculate_pd_total_net_input_wrt_weight(self,index_of_prev_neuron):#dZ/d(weight)
            return self.inputs[index_of_prev_neuron]
                  
 
class NeuronLayer(tf.keras.layers.Layer):
      def __init__(self,num_neurons,bias):
            self.num_neurons=num_neurons
            self.set_of_neurons=[]
            self.parameters={}
            # Every Layer shares the same bias
            self.bias=bias if bias else np.random.randn()

            for n in range(num_neurons):
                  self.set_of_neurons.append(Neuron(self.bias))

      def describe_neuron_layer(self):
            print('Number of Neurons : ',self.num_neurons)
            print('Neurons are :')
            for n in range(len(self.set_of_neurons)):
                  print('Neurons :', self.set_of_neurons[n])
                  for w in range(len(self.set_of_neurons[n].weights)):
                        print('Weigths of neuron',n,'are : ',self.set_of_neurons[n].weights[w])
                        self.parameters['W'+str(n)]=self.set_of_neurons[n].weights[w]
                        self.parameters['b'+str(n)]=self.bias
            print('Bias : ', self.bias)

            
      def feed_forward_layer(self,inputs):
            outputs=[]
            print('Inputs in  feed_forward_layer', inputs)

            for neuron in self.set_of_neurons:
                  #neuron.inputs=inputs

                  outputs.append(neuron.calculate_output(inputs))
            print('Output in feed forward layer :',outputs)

            return outputs
      

      def get_outputs(self,neuron_layer):
            outputs=[]
            for neuron in neuron_layer.set_of_neurons:
                  outputs.append(neuron.outputs)
            return outputs
      

class NeuralNetwork():
      def __init__(self,num_inputs,num_neurons_in_hidden_layer,num_neurons_in_outputs_layer,hidden_layer_weights=None,hidden_layer_bias=None,output_layer_weights=None,output_layer_bias=None):
            
            self.num_inputs=num_inputs
            self.hidden_layer=NeuronLayer(num_neurons_in_hidden_layer,hidden_layer_bias)
            print('Hidden layer',self.hidden_layer)
            self.output_layer=NeuronLayer(num_neurons_in_outputs_layer,output_layer_bias)
            self.LEARNING_RATE=0.5
            self.parameters={}
            self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
            self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)

      def get_params(self):
                  # Hidden layer parameters
                  for h_n in range(len(self.hidden_layer.set_of_neurons)):
                        for h_w in range(len(self.hidden_layer.set_of_neurons[h_n].weights)):
                              self.parameters[f'hidden_layer_W{h_n}_{h_w}'] = self.hidden_layer.set_of_neurons[h_n].weights[h_w]

                  # Hidden layer bias
                  self.parameters['hidden_layer_bias'] = self.hidden_layer.bias

                  # Output layer parameters
                  for o_n in range(len(self.output_layer.set_of_neurons)):
                        for o_w in range(len(self.output_layer.set_of_neurons[o_n].weights)):
                              self.parameters[f'output_layer_W{o_n}_{o_w}'] = self.output_layer.set_of_neurons[o_n].weights[o_w]

                  # Output layer bias
                  self.parameters['output_layer_bias'] = self.output_layer.bias

                  return self.parameters
      
      def create_batch(self,x_train,y_train,batch_size):
                  mini_batches=[]
                  data=np.stack((x_train,y_train),axis=1)
                  np.random.shuffle(data)# ensure that the mini-batches are representative of the overall dataset and not biased by the order of the original data
                  nber_of_batches=x_train.shape[0]//batch_size# number of complete mini-batches that can be created from the data.
                  print('NUMBER OF BATCH', nber_of_batches)
                  for i in range(nber_of_batches):
                        mini_batch=data[i*batch_size:(i+1)*batch_size]
                        mini_batches.append((mini_batch[:,0],mini_batch[:,1]))
                  if x_train.shape[0] % batch_size!=0:# Checks if there are remaining samples that do not fit into complete mini-batches. If true, an additional mini-batch with the remaining samples is created and added to mini_batches
                        print('THERE WAS REST')
                        mini_batch=data[i*batch_size:]
                        mini_batches.append((mini_batch[:,0],mini_batch[:,1]))
                  #print('#######################Returned mini batch',mini_batches)
                  return mini_batches
                  



      def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):# this parameter is a predefined list of weights for this hidden layer.
        #If this list is provided, the method will use these weights; otherwise, it will initialize random weights.
        
        weight_index = 0
        for h_n in range(len(self.hidden_layer.set_of_neurons)):
            for i in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layer.set_of_neurons[h_n].weights.append(random.random())
                    #print('Weights in init : ',self.hidden_layer.set_of_neurons[h_n].weights)
                else:
                    self.hidden_layer.set_of_neurons[h_n].weights.append(hidden_layer_weights[weight_index])
                weight_index += 1

      def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self,output_layer_weights):
            weight_index=0
            for o_n in range(len(self.output_layer.set_of_neurons)):
                  for h in range(len(self.hidden_layer.set_of_neurons)):
                        if not output_layer_weights:
                              self.output_layer.set_of_neurons[o_n].weights.append(random.random())
                        else:
                              self.output_layer.set_of_neurons[o_n].weights.append(output_layer_weights[weight_index])
                        weight_index += 1
            """
      def train_batch(self, batch_inputs, batch_outputs):
        for t in range(len(batch_inputs)):
            training_inputs, training_outputs = batch_inputs[t], batch_outputs[t]
            self.feed_forward_neural_network(training_inputs)
            self.backpropagate(training_outputs)
                     """

      def sigmoid_derivative(self,X):
            return X*(1-X)

      def feed_forward_neural_network(self,inputs):
            print('X_Batch in feed nn ',inputs)
            hidden_layer_outputs = self.hidden_layer.feed_forward_layer (inputs)
            return self.output_layer.feed_forward_layer(hidden_layer_outputs)
     
     
     
      def train_batch(self, training_inputs, training_outputs, epochs, batch_size):
            self.m=np.random.randn(1,1) # just one dependant variable. if 2 then (2,2)
            self.c=np.random.randn(1,1)
            l=len(training_inputs)
            
           
            for epoch in range(epochs):
                  batches=self.create_batch(training_inputs,training_outputs,batch_size)
                  for batch in batches:
                        print('Content of X',batch)
                        batch=np.append( np.array(batch[0][0]), np.array(batch[1][0])).reshape(1,3)
                        print('size of X',batch.size)
                        batch_normalization=BatchNormalization(X=batch, gamma=np.ones(3),beta=np.ones(3),bn_param={'mode': 'train'})
                        print('Content of X',batch)
                        normalized_batch,cache=batch_normalization.batchnorm_forward()
                        print('Normilized Batch',normalized_batch)
                        X_batch=normalized_batch[:,:2]
                        y_batch=normalized_batch[:,2]
                        print('X_batch Final', X_batch)
                        X_batch=X_batch.reshape(-1)
                        print('X_batch Final 2', X_batch)

                        # Forward and backward pass
                        self.feed_forward_neural_network(X_batch)
                        self.backward_neural_network(X_batch, y_batch)

                  
                  if epoch % 50 == 0:
                        #print('Epoch :', epoch, 'Total Loss :', round(self.calculate_total_error(training_inputs), 9),
                              self.describe_neural_network()
                  #print('TYPE HERE:',type(self.feed_forward_neural_network(X_batch)))
                  mse = np.mean(np.square(y_batch - self.feed_forward_neural_network(X_batch)))
                  print(f"Epoch {epoch + 1}/{epochs}, Mean Squared Error: {mse}")
            return training_inputs



      
      
      def backward_neural_network(self, X_batch, y_batch):
        self.feed_forward_neural_network(X_batch)

        # 1. Output neuron deltas
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.set_of_neurons)
        for o in range(len(self.output_layer.set_of_neurons)):

            # ∂E/∂zⱼ
            pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.set_of_neurons[o].calculate_pd_error_wrt_total_net_input(y_batch[o])

        # 2. Hidden neuron deltas
        pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(self.hidden_layer.set_of_neurons)
        for h in range(len(self.hidden_layer.set_of_neurons)):

            # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
            # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
            d_error_wrt_hidden_neuron_output = 0
            if self.output_layer.set_of_neurons:
                  for o in range(len(self.output_layer.set_of_neurons)):
                        d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.set_of_neurons[o].weights[h]

                        # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
                        pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * self.hidden_layer.set_of_neurons[h].calculate_der_output_neuron_wrt_Z()

        # 3. Update output neuron weights
        for o in range(len(self.output_layer.set_of_neurons)):
            for w_ho in range(len(self.output_layer.set_of_neurons[o].weights)):

                # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.set_of_neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.output_layer.set_of_neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight

        # 4. Update hidden neuron weights
        for h in range(len(self.hidden_layer.set_of_neurons)):
            for w_ih in range(len(self.hidden_layer.set_of_neurons[h].weights)):

                # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * self.hidden_layer.set_of_neurons[h].calculate_pd_total_net_input_wrt_weight(w_ih)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.hidden_layer.set_of_neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight


      def calculate_total_error(self, training_sets):
                  total_error = 0
                  for t in range(len(training_sets)):
                        training_inputs, training_outputs = training_sets[t]
                        self.feed_forward(training_inputs)
                        for o in range(len(training_outputs)):
                              total_error += self.output_layer.set_of_neurons[o].calculate_error(training_outputs[o])
                  return total_error
            
      def describe_neural_network(self):
                print('------')
                print('* Numbers of Inputs: {}'.format(self.num_inputs))
                print('------')
                print('Hidden Layer')
                self.hidden_layer.describe_neuron_layer()
                print('------')
                print('* Output Layer')
                self.output_layer.describe_neuron_layer()
                print('------')


def main():
      import random
      training_sets = np.array([
     [[-1, -1], [-1]],
     [[-1, 1], [1]],
     [[1, -1], [1]],
     [[1, 1], [-1]]
      ],dtype=object
                      )
      #training_inputs, training_outputs = random.choice(training_sets)
      training_inputs,training_outputs=training_sets[:,0], training_sets[:,1]
      print('******* Provided data in Main ', 'Inputs :',training_inputs,'Outputs:',training_outputs)

      nn = NeuralNetwork(len(training_sets[0][0]), 5, len(training_sets[0][1]))
      tr_inp= nn.train_batch(training_inputs,training_outputs,100,batch_size=1)
      print('Return of train_batch',tr_inp)
                
      for i in np.arange(len(nn.output_layer.set_of_neurons)):
             print('Outputs Neuron :',nn.output_layer.set_of_neurons[i].outputs)
      print('parameters are :', nn.get_params())
       
if __name__ == "__main__":
    main()