import numpy as np

# Fully Connected Layer
class Dense:
    def __init__(self, in_dim, out_dim, function, deriv_function):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.function = function
        self.deriv_function = deriv_function
        
        self.W = np.random.uniform(-0.08, 0.08, size=(in_dim, out_dim))
        self.b = np.zeros(shape=(out_dim,))
        self.delta = None
        
        self.h = None
        self.u = None
        self.dW = None
        self.db = None
        
    def __call__(self, h: np.ndarray):
        # h: (batch_size, in_dim)
        # next_h: (batch_size, out{j}_dim)
        self.h = h
        self.u = h @ self.W + self.b     # calculate next layer
        next_h = self.function(self.u)   # Apply activation
        return next_h
    
    def b_prop(self, delta: np.ndarray, W: np.ndarray):
        """
        delta_{j}^{l} = sum_k delta_{k}^{l+1} * du_{k}^{l+1}/dj^l
        """
        # delta: (batch_size, out{j+1}_dim)
        # W: (out{j}_dim, out{j+1}_dim)
        self.delta = self.deriv_function(self.u) * (delta @ W.T)
        return self.delta
        
    def compute_grad(self):
        """
        
        """
        # self.h: (batch_size, in_dim)
        # self.delta: (batch_size, out{j}_dim)
        # self.dW: (in_dim, out{j}_dim)
        # self.db: (out{j}_dim)
        
        # Calculate batch size to divide the derivative to average
        batch_size = self.h.shape[0]
        self.dW = self.h.T @ self.delta / batch_size
        self.db = np.ones(batch_size) @ self.delta / batch_size
    

class Model:
    def __init__(self, hidden_dims, activation_functions, deriv_functions):
        """
        Parameters
        ----------
        hidden_dims: list of width of each layers & input (n+1)
        activation_functions: list that stores function for each layer (n)
        deriv_functions: list that stores derivative of functions for each layer (n)
        """
        # Create each dense network
        self.layers = []
        for i in range(len(hidden_dims) - 1):
            self.layers.append(Dense(hidden_dims[i], hidden_dims[i+1], activation_functions[i], deriv_functions[i]))
    
    
    def __call__(self, x):
        return self.forward(x) 
    
    def forward(self, x):
        # Simply multiply & pass on
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, delta):
        # output layer
        self.layers[-1].delta = delta
        self.layers[-1].compute_grad()
        W = self.layers[-1].W
        # other layers
        for layer in self.layers[-2::-1]:
            delta = layer.b_prop(delta, W)
            layer.compute_grad()
            W = layer.W
    
    def update(self, eps=0.01):
        for layer in self.layers:
            layer.W -= eps * layer.dW
            layer.b -= eps * layer.db
