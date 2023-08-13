# Client Valuation and Selection for Communication-Efficient Federated Learning
This repository implements the **Fed-Shap-UCB** algorithm for accelerating convergence in federated learning and compares it against other baselines like **FedAvg**, **FedProx**, **S-FedAvg**, **Power-Of-Choice** and **Centralised** training on the **MNIST**, **FMINST**, **CIFAR-10** and **Synthetic** datasets.

***

To run these algorithms execute main.py with the desired settings (edit the file).\
(TODO: implement argparse to accept comand-line arguments)

Dataset Configuration:
1. name of dataset (from ```['fmnist','cifar10','mnist','synthetic']```)
2. number of clients ($N$, any positive integer)
3. alpha ($\alpha$ parameter for dirichlet distribution, not required for Synthetic dataset) (typically varied in powers of 10 from $10^{-4}$ to $10^4$)
4. alpha, beta (parameters for the Synthetic dataset only) (typically $\alpha = \beta = 0, 0.5, 1$)

Algorithm Configuration:
1. algorithms to execute (from '''['fedavg','fedprox','sfedavg','ucb','centralised','poc']''')
2. client select fraction $\frac{M}{N}$
3. E, B, lr, momentum (epochs, batches, learning rate, SGD momentum)
4. T (number of communication rounds)
5. noise level (maximum client update noise in the privacy preserving setting)

Algorithm Hyperparameters
1. S-FedAvg (we vary $\alpha = 1- \beta$)
2. Power-Of-Choice (decay factor $\lambda$)
3. FedProx (weight of proximal term $\mu$)
4. Fed-Shap-UCB (exploration-exploitation tradeoff $\beta$)

Logging results: \\
if logging is set to True the runs are saved locally in pickle objects and and on wandb \
(TODO: make it easier to set file location for logging runs locally)

You can set the above parameters to a single value or implement a hyperparameter sweep over a list of values. After setting the desired values simply execute the following
```
python main.py
```

### plotting.py
In order to visualise results we download them from wandb into a Pandas Dataframe and display the desired metrics using the Seaborn plotting library. Some results are already in the *plots* folder\
(TODO: restructure file to generate any of the finalised plots from the command line without editing the code for reproducibility)

### server.py
Implements the Server class with methods for:
1. client model aggregation
2. Shapley Value computation
3. returning server model performance metrics (accuracy and loss)

4 different kinds of Shapley Value estimation have been implemented:
1. Monte Carlo sampling
2. Truncated Monte Carlo sampling
3. GTG-Shapley (used by our method)
4. True Shapley Value (extremely expensive, computes loss over all subsets)

Convergence criterion for Shapley Values is implemented in ```utils.py```

### client.py
Implements the Client class with methods for:
1. client model training
2. adding noise to updates
3. returning client model performance metrics on client data (accuracy and loss)

### algorithms.py
Implements all the above mentioned Federated Learning algorithms. Every method returns ```test_acc, train_acc, train_loss, val_loss, test_loss, selections```.\
FedProx and FedAvg loss are defined using nested functions (closures). The returned loss functions have a slightly different signature from loss functions in PyTorch.

### data_preprocess.py
Implements methods for downloading and splitting datasets into train-val-test, generating synthetic data and splitting data across clients using the power law and Dirichlet distribution.

### initialise.py
Constructs server object and desired number of client objects with data and models allocated to each of them.

### model.py
Implements two different models: a Multi-Layer-Perceptron (NN) and a Convolutional Neural Network (CNN)

### utils.py
implements some utility functions

