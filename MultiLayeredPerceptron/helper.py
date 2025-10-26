import numpy as np

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

# Training model
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

# Validation of model
def valid(model, x, t):
    y = model(x)
    loss = cross_entropy_loss(t, y)
    # No need to update
    return loss, y