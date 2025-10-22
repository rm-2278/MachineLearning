import numpy as np


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

