# Neural Network with optimization
Implements neural network with Pytorch. Uses regularizers and optimizers to improve the performance.

To use it:
1. python -m venv venv
2. source venv/bin/activate (Linux/Bash/mac)
   .\venv\Scripts\activate (Windows Command Prompt)
3. pip install -r requirements.txt
4. wandb login -> fill in api
5. python main.py


Improvements from previous models:
- Uses schedular (Adam)
- Uses wandb for logging
- Uses nn.CrossEntropyLoss, which is more suitable than F.cross_entropy as former is for within models while latter is for one-time. Also, this internally does equivalent calculation as one-hot encoding so removes the code for manually transforming.
- Uses GPU if cuda exists
- Uses main guard
- Reuses evaluate function for validataion & testing


fashion_mnist.py:
Uses the model for the fashion_mnist task, with 10 categories.
With 10 epochs,

train_loss: 0.5378 -> 0.2511

valid_loss: 0.3923 -> 0.2943

train_acc: 0.8112 -> 0.9048

valid_acc: 0.8587 -> 0.8937

test_loss: 0.3320

test_acc: 0.8831

Private log:
https://wandb.ai/rm2278-university-of-cambridge/NNO_fashio_mnist/

TODO:
1. Find the optimal hyperparameters using grid/random search
2. Train for longer
3. Make model deeper
4. Apply to other tasks

