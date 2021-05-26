import numpy
import numpy as np

from copy import copy


class RCN:
    def __init__(self
                 , n_reservoir=100
                 , n_inputs=1
                 , n_outputs=1
                 , spectral_radius=0.95
                 , input_scaling=1
                 , bias=0
                 , constant_bias=True
                 , sparsity=0
                 , leak_rate=1
                 , noise=0
                 , reservoir_type="normal"
                 , activation="tanh"
                 , read_out_type="linear"
                 , optimization="ridge"
                 , regularization=0
                 , force_sr=True
                 , distribute_input=False
                 , transient=100
                 , random_state=None
                 ):

        '''

        Parameters
        ----------
        n_reservoir : int, default = 100
            Number of neurons in the reservoir.
        n_inputs : int, default = 1
            dimensionality of the input.
        n_outputs: int, default = 1
            dimensionality of the output.
        spectral_radius : float, default = 0.95
            spectral radius of the reservoir.
            The spectral radius is defined as the largest absolute value of the eigenvalues of a (square) matrix.
        input_scaling: float, default = 1
            The scaling of the input matrix element.
        bias: float, default = 1
            A bias term which adds up to the value of the pre-activation
        sparsity: float, default = 0
            fraction of null element in the reservoir matrix.
        leak_rate: float, default = 1
            controls leakage of the system.
            When `leak_rate` = 1 The system has no leakage, when =0 the system does not update.
        noise: float, default = 0
            Magnitude of the noise introduced into the reservoir.
        reservoir_type: string, default = "normal"
            Controls the way the reservoir is constructed.
            - "normal"
            - "wigner"
            - "ring"
            - "delay_line"
        activation: string, default = "tanh"
            Controls the kind of activation for the reservoir:
            - "tanh"
            - "linear"
        read_out_type: string, default = "linear"
            Selects the type of readout the network use:
            - "linear"
            - "augmented"
            - "square"
        optimization: string, default = "ridge"
            Selects the optimization used to feat the readout.
            - "ridge"
        regularization: default = 0
            Regularization paremeter(s) of the regularization algorithm.
        force_sr: bool, default = True
            If `True` the spectral radius is exactly the value given in input.
            Otherwise, its expected value will be the input value, but they may not match exactly.
        distribute_input: bool, default = False
            If true, only on element of each row of the input vector is non-null.
        transient: int, default = 100
            The number of input values to discard when fitting.
        random_state:
        '''

        self.transient = transient
        self.regularization = regularization
        self.optimization = optimization
        self.n_reservoir = n_reservoir
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.constant_bias = constant_bias

        if constant_bias:
            self.bias = bias
        else:
            self.bias = ((np.random.rand(self.n_reservoir) * 2) - 1) * bias

        self.leak_rate = leak_rate
        self.sparsity = sparsity
        self.noise = noise

        self.force_sr = force_sr
        self.distribute_input = distribute_input

        # the given random_state might be either an actual RandomState object,
        # a seed or None (in which case we use numpy's builtin RandomState)
        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: " + str(e))
        else:
            self.random_state_ = np.random.mtrand._rand

        assert activation in {"tanh", "linear", "relu", "sin"}, "Possible values: {'tanh', 'linear', 'relu', 'sin'} \'."
        if activation == "tanh":
            self.activation = np.tanh

        elif activation == "linear":
            self.activation = lambda x: x

        elif activation == "relu":
            self.activation = lambda x: x * (x > 0)

        elif activation == "sin":
            self.activation = np.sin



        assert reservoir_type in {"normal", "wigner", "ring", "delay_line"}, \
            'Possible values: {"normal", "wigner", "ring", "delay_line"} \'.'
        self.reservoir_type = reservoir_type

        assert read_out_type in {"linear", "augmented", "square"}, \
            'Possible values: {"linear", "augmented", "square"} \'.'
        self.read_out_type = read_out_type

        self._create_net()

        self.states = None
        self.last_state = None
        self.last_input = None
        self.last_output = None

    def _create_net(self):
        self._create_reservoir()
        self._create_read_in()

    def _create_reservoir(self):
        if self.reservoir_type == "normal":
            # Weights created using a normal distribution
            W = self.random_state_.randn(self.n_reservoir, self.n_reservoir)
            # sparify the matrix
            W[self.random_state_.randn(self.n_reservoir, self.n_reservoir) < self.sparsity] = 0

        elif self.reservoir_type == "wigner":
            # Weights created using a normal distribution
            W = self.random_state_.randn(self.n_reservoir, self.n_reservoir)
            # sparify the matrix
            W[self.random_state_.randn(self.n_reservoir, self.n_reservoir) < self.sparsity] = 0
            # use the symmetrization
            W = (W + W.T) // 2

        elif self.reservoir_type == "ring":
            W = np.eye(self.n_reservoir, k=-1)
            W[0, self.n_reservoir - 1] = 1

        elif self.reservoir_type == "delay_line":
            W = np.eye(self.n_reservoir, k=-1)

        if self.force_sr:
            # normalized the spectral radius so that it equals exactly the given value
            old_radius = np.max(np.abs(np.linalg.eigvals(W)))
            W *= (self.spectral_radius / old_radius)

        else:
            # this uses the expected value instead
            W *= (self.spectral_radius / np.sqrt((1.0 - self.sparsity) * self.n_reservoir))

        self.W_res = W

    def _create_read_in(self):
        # W_in in created using normal distribution
        if self.distribute_input:
            W_in = np.zeros(self.n_reservoir, self.n_inputs)
            for i in range(W_in.shape[0]):
                j = np.random.randint(0, W_in.shape[1])
                W_in[i, j] = np.random.randn()

        else:
            W_in = self.random_state_.randn(self.n_reservoir, self.n_inputs)

        self.W_in = self.input_scaling * W_in

        self.read_in = lambda x: W_in.dot(x)

    def _update(self, state, input):

        preactivation = self.W_res.dot(state) + self.read_in(input) + self.bias
        if self.noise != 0:
            noise_term = self.noise * (self.random_state_.randn(self.n_reservoir))
            preactivation += noise_term

        new_state = self.leak_rate * self.activation(preactivation) + (1. - self.leak_rate) * state

        return new_state

        # This is only the evolution, with no training

    def train(self, inputs, outputs):
        self.listen(inputs)
        return self.fit(outputs)

    def listen(self, inputs):

        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (-1, inputs.shape[0]))
        T_train = len(inputs[0])
        # generate the entire sequence of states
        states = np.zeros((self.n_reservoir, T_train + 1))
        for t in range(T_train):
            states[:, t + 1] = self._update(states[:, t], inputs[:, t])

        self.states = states[:, :-1]
        self.last_state = self.states[:, -1]
        self.last_input = inputs[:, -1]

    def fit(self, outputs, states=None):

        if states is None:
            states = self.states
        if outputs.ndim < 2:
            outputs = np.reshape(outputs, (-1, len(outputs)))

        if self.read_out_type == "linear" and self.optimization == "ridge":
            # use the transpose as the most common form is Ax = b
            # while we have WR=Y
            A = self.states[:, self.transient:].T
            b = outputs[:, self.transient:].T

            # compute the two terms
            first = A.T @ b
            second = np.linalg.pinv(A.T @ A + self.regularization * np.identity(A.shape[1]))
            self.W_out = (second @ first).T  # transpose is needed now to obtain the correct W
            # define the readout function
            self.read_out = lambda x: self.W_out.dot(x)

        if self.read_out_type == "augmented" and self.optimization == "ridge":
            # use the transpose as the most common form is Ax = b
            # while we have WR=Y

            R_aug = self._augment_state(self.states[:, self.transient:])
            A = R_aug.T
            b = outputs[:, self.transient:].T

            # compute the two terms
            first = A.T @ b
            second = np.linalg.pinv(A.T @ A + self.regularization * np.identity(A.shape[1]))
            self.W_out = (second @ first).T  # transpose is needed now to obtain the correct W
            # define the readout function
            self.read_out = lambda x: self.W_out.dot(self._augment_state(x))

        if self.read_out_type == "square" and self.optimization == "ridge":
            # use the transpose as the most common form is Ax = b
            # while we have WR=Y

            R_sq = self._square_state(self.states[:, self.transient:])
            A = R_sq.T
            b = outputs[:, self.transient:].T

            # compute the two terms
            first = A.T @ b
            second = np.linalg.pinv(A.T @ A + self.regularization * np.identity(A.shape[1]))
            self.W_out = (second @ first).T  # transpose is needed now to obtain the correct W
            # define the readout function
            self.read_out = lambda x: self.W_out.dot(self._square_state(x))

        self.last_output = outputs[:, -1]

        return self.read_out(states)

    def predict(self, inputs, continuation=True):
        """
        Apply learned model on test data.

        :param inputs: array of dimensions (N_test_samples x n_inputs)
        :param continuation: if True, start the network from the last training state
        :return predictions on test data
        """

        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (-1, len(inputs)))

        # set noise term to zero for state update during test
        self.noise = 0

        T_test = inputs.shape[-1]

        if continuation:
            last_state = self.last_state
            last_input = self.last_input
            last_output = self.last_output
        else:
            last_state = np.zeros(self.n_reservoir)
            last_input = np.zeros(self.n_inputs)
            last_output = np.zeros(self.n_outputs)

        inputs = np.hstack([last_input.reshape(-1, 1), inputs])
        states = np.hstack([last_state.reshape(-1, 1), np.zeros((self.n_reservoir, T_test))])
        outputs = np.hstack([last_output.reshape(-1, 1), np.zeros((self.n_outputs, T_test))])

        # process test set one sample at a time
        for t in range(T_test):
            # next state
            states[:, t + 1] = self._update(states[:, t], inputs[:, t])
            # predicted output
            outputs[:, t + 1] = self.read_out(states[:, t + 1])

        # stack up new states
        self.states = np.hstack((self.states, states[:, 1:]))

        return outputs[:, 1:]

    def _augment_state(self, R):
        if R.ndim < 2:
            R = R.reshape(-1, 1)

        R_aug = copy(R)
        R_aug[::2, :] = R_aug[::2, :] ** 2
        return R_aug

    def _square_state(self, R):
        """
        Parameters
        ----------
        R

        Returns
        -------

        """
        if R.ndim < 2:
            R = R.reshape(-1, 1)

        R_sq = np.vstack((copy(R), R ** 2))

        return R_sq

    def get_augmented_states(self):
        return self._augment_state(self.states)

    def get_square_states(self):
        return self._square_state(self.states)


class AutonomousRCN(RCN):

    def __init__(self, n_reservoir=100
                 , n_inputs=1
                 , spectral_radius=0.95
                 , input_scaling=1
                 , bias=0
                 , constant_bias=True
                 , sparsity=0
                 , leak_rate=1
                 , noise=0
                 , reservoir_type="normal"
                 , activation="tanh"
                 , read_out_type="linear"
                 , optimization="ridge"
                 , training_type="off_line"
                 , regularization=0
                 , force_sr=True
                 , distribute_input=False
                 , transient=100
                 , random_state=None
                 ):
        """

        Parameters
        ----------
        n_reservoir
        n_inputs
        spectral_radius
        input_scaling
        bias
        constant_bias
        sparsity
        leak_rate
        noise
        reservoir_type
        activation
        read_out_type
        optimization
        training_type
        regularization
        force_sr
        distribute_input
        transient
        random_state
        """



        super(AutonomousRCN, self).__init__(n_reservoir=n_reservoir
                                            , n_inputs=n_inputs
                                            , n_outputs=0
                                            , spectral_radius=spectral_radius
                                            , input_scaling=input_scaling
                                            , bias=bias
                                            , constant_bias=constant_bias
                                            , sparsity=sparsity
                                            , leak_rate=leak_rate
                                            , noise=noise
                                            , reservoir_type=reservoir_type
                                            , activation=activation
                                            , read_out_type=read_out_type
                                            , optimization=optimization
                                            , regularization=regularization
                                            , force_sr=force_sr
                                            , distribute_input=distribute_input
                                            , transient=transient
                                            , random_state=random_state
                                            )

        self.training_type = training_type

    def train(self, inputs:numpy.array):
        """
        Parameters
        ----------
        inputs
        """
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (-1, inputs.shape[0]))

        if self.training_type == "off_line":
            self.listen(inputs[:, :])
            return self.fit(inputs[:, :])

    def predict(self, n_steps, continuation=True):
        """
        Parameters
        ----------
        n_steps
        continuation

        Returns
        -------

        """

        # set noise term to zero for state update during test
        self.noise = 0

        T_test = n_steps

        if continuation:
            last_state = self.last_state
            last_input = self.last_input
            last_output = self.last_output
        else:
            last_state = np.zeros(self.n_reservoir)
            last_input = np.zeros(self.n_inputs)
            last_output = np.zeros(self.n_outputs)

        # next_input = self.read_out(last_state)

        # estimated inputs to be computed
        # the first is the last "true" inputs
        est_inputs = np.hstack(
            [last_input.reshape(-1, 1), np.zeros((self.n_inputs, T_test))])

        states = np.hstack([last_state.reshape(-1, 1), np.zeros((self.n_reservoir, T_test))])
        # outputs = np.hstack([last_output.reshape(-1, 1), np.zeros((self.n_outputs, T_test))])

        for t in range(T_test):
            # next state
            states[:, t + 1] = self._update(states[:, t], est_inputs[:, t])
            # predicted estimated inputs
            est_inputs[:, t + 1] = self.read_out(states[:, t + 1])

        # stack up new states
        self.states = np.hstack((self.states, states[:, 1:]))

        return est_inputs[:, 1:]
