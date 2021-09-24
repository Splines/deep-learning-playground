import logging
from math import sqrt

import numpy as np

from helper import encode_one_hot, normalize, relu, sigmoid, softmax

logging.basicConfig(filename='nn.log', filemode='w',
                    encoding='utf-8', level=logging.INFO)


class NN:

    def __init__(self, layers, batch_size, step_size):
        """
        Inits the neural network.

        Parameters:
            layers: [ layer1_n, layer2_n, layer3_n, ... ]
            batch_size: the size of the batch
            step_size: factor the gradients are multiplied with
        """
        # Set metadata
        if not len(layers) >= 2:
            raise TypeError(
                "Specify at least two layers for the neural network")
        self.layer_sizes = layers
        self.batch_size = batch_size
        self.count = 0
        self.cost = 0
        self.step_size = step_size

        # Init neurons
        self.neurons_z = [[]] + [np.empty((n, 1), dtype='float64')
                                 for n in self.layer_sizes[1:]]
        self.neurons = [np.empty((n, 1), dtype='float64')
                        for n in self.layer_sizes]

        # Init weights & biases
        # https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
        # He Weight initialization
        standard_deviation = sqrt(2.0 / self.layer_sizes[0])
        self.weights = [[]] + [np.random.randn(self.layer_sizes[i+1], k).astype('float64') * standard_deviation
                               for i, k in enumerate(self.layer_sizes[:-1])]
        # biases are initialized by default to 0
        self.biases = [[]] + [np.zeros((n, 1), dtype='float64')
                              for n in self.layer_sizes[1:]]
        self.log_state()

        # Init desired changes for backpropagation
        self.weights_desired_changes = [[]] + \
            [np.empty((self.layer_sizes[i+1], n2), dtype='float64')
             for i, n2 in enumerate(self.layer_sizes[:-1])]
        self.biases_desired_changes = [[]] + \
            [np.empty((n, 1), dtype='float64') for n in self.layer_sizes[1:]]

    def log_state(self):
        logging.info('###### State information ######')
        logging.info('--- Weights')
        logging.info(self.weights)
        logging.info('--- Biases')
        logging.info(self.biases)

    def train(self, input, desired):
        """Trains the neural net with one sample."""
        # more desired values than output neurons
        if desired >= self.layer_sizes[-1]:
            raise ValueError(
                f"Desired value is {desired} but must be smaller than {len(self.neurons[-1])}")

        # --- Feed forward & Propagate back
        output = self._feed_forward(input)
        label_prediction = np.argmax(output)
        is_prediction_correct = (label_prediction == desired)

        desired = encode_one_hot(desired, len(output))
        # Propagate back (reset desired changes at start of new batch)
        self._propagate_back(
            desired, should_reset_desired_changes=self.count == 0)

        # Calc cost
        # self.cost += np.sum((output-desired)**2)
        self.cost += np.sum(np.square(output-desired))

        # Learn after <batch_size> trainings
        self.count += 1
        if self.count == self.batch_size:
            # Get & reset cost
            self.cost /= self.batch_size  # average
            # print(f"C={self.cost}")
            cost_tmp = self.cost
            self.cost = 0

            # Learn
            self._learn()
            self.log_state()

            # Reset batch
            self.count = 0

            return (is_prediction_correct, cost_tmp)

        return (is_prediction_correct, None)

    def _feed_forward(self, input):
        """Feeds forward through neural network. Returns the output neurons."""
        # Checks
        if not isinstance(input, np.ndarray):
            raise TypeError(f"Input must be a ndarray")
        if not input.size == self.layer_sizes[0]:
            raise TypeError(
                f"Input contains {input.size} entries, however the input layer is expecting {len(self.neurons[0])} entries")

        # Set input layer
        self.neurons[0] = input.reshape(input.shape[0], 1)

        # Feed forward through all layers, start with layer l=1
        for l in range(1, len(self.neurons)):
            # print(f'weights: {self.weights[l].shape} @ {self.neurons[l-1].shape} + {self.biases[l].shape}')
            # add two column vectors together
            x = self.weights[l] @ self.neurons[l-1]
            self.neurons_z[l] = x + self.biases[l]
            self.neurons[l] = self._activation(self.neurons_z[l])

        # Last layer: add softmax to get probabilities in range [0,1]
        # self.neurons[-1] = softmax(self.neurons[-1])

        # Return neurons of output layer
        return softmax(self.neurons[-1])

    def _propagate_back(self, desired, should_reset_desired_changes=False):
        """
        Propagates back the results and saves desired changes to weights and biases.
        Note that we do not make any changes to weights or biases at this point yet (this is done in _learn()).
        """
        # Go backwards through the layers and calculate gradient
        delC_delActivation = None
        delZ_delWeights = None

        # from last layer to layer 1 (not: to layer 0 (!))
        for l in range(len(self.neurons)-1, 0, -1):
            # Calc delC_delActivation
            if l == len(self.neurons)-1:  # output layer
                delC_delActivation = 2*(self.neurons[l]-desired)
            else:
                inner_prod = delC_delActivation * \
                    self._v_derive_activation(self.neurons_z[l+1])

                weights_j = [self.weights[l+1][:, j]
                             for j in range(self.layer_sizes[l])]
                weights_j = [weights_m.reshape(
                    (weights_m.shape[0], 1)) for weights_m in weights_j]

                delC_delActivation = np.array([
                    sum(inner_prod * weights_m) for weights_m in weights_j
                ])

            # Calc delActivation_delZ
            delActivation_delZ = self._v_derive_activation(self.neurons_z[l])
            biases = delC_delActivation * delActivation_delZ  # column vector

            # Calc delZ_delWeights
            delZ_delWeights = self.neurons[l-1]  # column vector

            # Calc delC_delWeights
            delC_delWeights = biases @ delZ_delWeights.T

            # Add to desired weights & biases
            if should_reset_desired_changes:
                self.weights_desired_changes[l] = delC_delWeights
                self.biases_desired_changes[l] = biases
            else:
                self.weights_desired_changes[l] += delC_delWeights
                self.biases_desired_changes[l] += biases

    def _learn(self):
        """Does a gradient step after having gone through a mini batch"""
        # Go through layers (except input layer)
        for l in range(1, len(self.neurons)):
            # Get desired changes of weights & biases
            # (average over the desired changes of each sample in the mini batch)
            self.weights_desired_changes[l] /= self.batch_size  # average
            self.weights_desired_changes[l] *= self.step_size  # step size
            self.weights_desired_changes[l] *= -1  # negative gradient

            self.biases_desired_changes[l] /= self.batch_size  # average
            self.biases_desired_changes[l] *= self.step_size  # step size
            self.biases_desired_changes[l] *= -1  # negative gradient

            # Adjust weights & biases to "learn"
            self.weights[l] += self.weights_desired_changes[l]
            self.biases[l] += self.biases_desired_changes[l]

    def _activation(self, vector):
        # ReLu
        activated = [relu(x) for x in vector]
        return np.array(activated, ndmin=2)  # col vector

    def _derive_activation(self, x):
        """Calculates the derivative of the activation function"""
        if x > 0:
            return 1
        else:
            return 0

    def _v_derive_activation(self, x):
        """Vectorized version of _derive_activation"""
        return np.vectorize(self._derive_activation)(x)
