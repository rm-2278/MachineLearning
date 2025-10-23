import numpy as np

class LogisticRegression:
    def __init__(self, input_dim, num_classes):
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Initialize weights and biases
        self.W = np.random.uniform(low=-0.08, high=0.08, size=(input_dim, num_classes)).astype('float32')
        self.b = np.zeros(shape=(num_classes,)).astype('float32')
        
    def softmax(self, x: np.ndarray):
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
        x -= x.max(axis=1, keepdims=True) # Prevent exp overflow
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def train(self, x: np.ndarray, t: np.ndarray, eps=0.1):
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
        pred = self.softmax(x @ self.W + self.b) # (batch_size, num_classes)
        loss = -np.mean(np.sum(t * np.log(np.clip(pred, 1e-8, 1.0)), axis=1)) # Add epsilon for stability
        
        # Calculate gradient
        dW = -(x.T @ (t - pred)) / x.shape[0] # (input_dim, num_classes)
        dE = -np.mean(t - pred, axis=0) # (num_classes,)
        
        # Update parameter
        self.W -= eps * dW
        self.b -= eps * dE
        
        return loss

    def valid(self, x: np.ndarray, t: np.ndarray):
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
        pred = self.softmax(x @ self.W + self.b)
        loss = -np.mean(np.sum(t * np.log(np.clip(pred, 1e-8, 1.0)), axis=1))
        return loss, pred
