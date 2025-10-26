import numpy as np
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load dataset (train: 60000 * 28*28, test: 10000 * 28*28)
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Preprocessing (Normalization, one-hot encoding)
x_train, x_test = x_train / 255., y_train / 255.
x_train, x_test = x_train.reshape(x_train.shape[0], -1), x_test.reshape(x_test.shape[0], -1)
y_train = to_categorical(y_train, num_classes=10)

# Split training data
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)


# Split data into batches
def create_batch(data: np.ndarray, batch_size: int):
    """
    Parameters
    ----------
    data: (dataset_size, data_size)
    batch_size: 
    
    Returns:
    --------
    batched_data: (num_batches + (mod!=0), batch_size, data_size)
    """
    
    num_batches, mod = divmod(data.shape[0], batch_size)
    batched_data = np.split(data[:batch_size*num_batches], num_batches) # split main chunk
    if mod:
        batched_data.append(data[batch_size*num_batches:]) # Add the batch for the rest
    
    return batched_data

# Set random seed
random_state = 42
rng = np.random.RandomState(random_state)

# Activation function
def relu(x: np.ndarray):
    # x: (batch_dim, layer_dim)
    return np.maximum(x, 0)

# Derivative of ReLU
def deriv_relu(x: np.ndarray):
    # x: (batch_dim, layer_dim)
    return (x > 0).astype(x.dtype)
    
# Activation function (sum of p = 1)
def softmax(x: np.ndarray):
    # x: (batch_dim, layer_dim)
    x -= x.max(axis=1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

# Derivative of softmax
def deriv_softmax(x: np.ndarray):
    return softmax(x) * (1 - softmax(x))

# Calculate loss
def cross_entropy_loss(t:np.ndarray, y:np.ndarray):
    # t: True values (batch_size,)
    # y: Predicted values (batch_size, )
    return np.sum(-t * np.log(y), axis=1).mean()

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


lr = 0.01
epochs = 10
batch_size = 128



model = Model(hidden_dims=[784, 100, 100, 10], activation_functions=[relu, relu, softmax], deriv_functions=[deriv_relu, deriv_relu, deriv_softmax])


def train(model, x, t, eps=0.01):        
    """
    delta^{L} = dE(theta)/dh * dh/du
              = sum_{j} (- t{j}/y{j} * y{j} (delta{i}{j} - y{i})) = -t{i} + y{i} (since cross_entropy & softmax)
    """
    
    # forward pass
    y = model(x)
    
    # calculate loss
    loss = cross_entropy_loss(t, y)
    
    # back-propagation
    delta = y - t
    model.backward(delta)
    
    # update
    model.update(eps=eps)
    
    return loss,


def valid(model, x, t):
    y = model(x)
    loss = cross_entropy_loss(t, y)
    # No need to update
    return loss, y


for epoch in range(epochs):
    # Shuffle randomly to recreate batches
    x_train, y_train = shuffle(x_train, y_train)
    x_train_batch, y_train_batch = create_batch(x_train, batch_size), create_batch(y_train, batch_size)
    
    for x, t in zip(x_train_batch, y_train_batch):
        loss = train(model, x, t)
    
    loss, y_pred = valid(model, x_val, y_val)
    accuracy = accuracy_score(y_val.argmax(axis=1), y_pred.argmax(axis=1))
    if epoch == 0 or epoch % 2 == 1:
        print(f"EPOCH: {epoch+1} Valid Cost: {loss:.3f} Valid Accuracy: {accuracy:.3f}")