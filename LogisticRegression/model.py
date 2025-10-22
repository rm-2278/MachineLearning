import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Loading dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Scale the pixel value: 0 - 255 -> 0.0 - 1.0
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Flatten each image: 28 * 28 -> 784
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

# Convert integer label to one-hot vector
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Softmax function
def softmax(x: np.ndarray):
    """
    Compute the softmax of each row in the input array
    
    Parameters
    ----------
    x: np.ndarray, (batch_size, input_dim)
        Input array
    
    Returns
    -------
    np.ndarray, (batch_size, input_dim)
        Output array, where each row sums to 1
        and represents probability distribution.
    """
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

input_dim, num_classes = 784, 10

# Initialize weights and biases
W = np.random.uniform(low=-0.08, high=0.08, size=(input_dim, num_classes)).astype('float32')
b = np.zeros(shape=(num_classes,)).astype('float32')


# Split train dataset to train and validation
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1)


def train(x: np.ndarray, t: np.ndarray, eps=0.1):
    """
    Parameters
    ----------
    x: np.ndarray, (batch_size, input_dim)
    t: np.ndarray, (batch_size, num_classes)
    eps: float
    
    Returns
    -------
    score: float
        returns the average cross entropy loss
    """
    global W, b
    pred = softmax(x @ W + b) # (batch_size, num_classes)
    loss = -np.mean(np.sum(t * np.log(np.clip(pred, 1e-8, 1.0)), axis=1)) # Add epsilon for stability
    
    # Calculate gradient
    nablaW_E = -(x.T @ (t - pred)) / x.shape[0] # (input_dim, num_classes)
    nablab_E = -np.mean(t - pred, axis=0) # (num_classes,)
    
    # Update parameter
    W -= eps * nablaW_E
    b -= eps * nablab_E
    
    return loss


def valid(x: np.ndarray, t: np.ndarray):
    """
    Parameters
    ----------
    x: np.ndarray, (batch_size, input_dim)
    t: np.ndarray, (batch_size, num_classes)
    
    Returns
    -------
    score: float
        returns the average cross entropy loss
    """
    global W, b
    pred = softmax(x @ W + b)
    loss = -np.mean(np.sum(t * np.log(np.clip(pred, 1e-8, 1.0)), axis=1))
    return loss

# Training loop
for epoch in range(5):
    train_loss = train(x_train, y_train)
    valid_loss = valid(x_valid, y_valid)
    print(f'[Epoch {epoch+1}] train_loss: {train_loss} valid_loss: {valid_loss}')

# Evaluation
test_loss = valid(x_test, y_test)
print(f'Evaluation Loss: {test_loss}')