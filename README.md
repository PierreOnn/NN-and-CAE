How to run the code:

Besides **NeuralNetwork.py** with *sweeps*, another python file **NNtest.py** was created to 
illustrate the working of the architecture.
NNtest.py represents a predefined configuration of the learning rate and number of epochs
and computes the corresponding training accuracy.

### NNtest.py
This pyhton file could be run freely and serves to provide the inner working of the architecture.
Herewith, the learning rate and number of epochs are initialized to 0.5 and 110 respectively
for obtaining an accuracy of 100%.

### NeuralNetwork.py (with Sweeps)
If you're running the code with a sweep in NeuralNetwork.py, please execute the following steps in the terminal (https://wandb.ai/quickstart/pytorch):
- pip install wandb
- wandb login
- 3baf9835f7643ccdd5a2df4c3d89245c61479e0f
- wandb sweep sweep-grid.yaml
- wandb agent machinelearningassignment/MachineLearning/10u28ok3

Now, when logged in and the agent has started, it should be able to run the code in regard to 
the predefined sweeps in one of the yaml files and collect the outcomes on the platform.

Unfortunately because of the deep integration for hyperparameter tuning, it is not 
possible to run that code without Weights and Biases. 

As a reminder, if there is some trouble running the sweeps with Weights and biases,
our group refers to the running of the NNtest.py.
