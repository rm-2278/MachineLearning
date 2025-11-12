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
train_loss: 
valid_loss:
test_loss:
train_acc:
valid_acc:  
test_acc: 


TODO:
1. Find the optimal hyperparameters using grid/random search
2. Train for longer
3. Make model deeper
4. Apply to other tasks

