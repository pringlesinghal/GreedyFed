# GreedyFed: Greedy Shapley Selection for Communication-Efficient Federated Learning <a href="https://drive.google.com/file/d/19nMQb0LPI2YCVpalM9eZPBITOnXuQUFl/view?usp=sharing"><img src="docs/assets/file-earmark-fill.svg" alt="paper" width="24" height="24"></a>
This repository implements the **GreedyFed** algorithm for accelerating convergence in federated learning and compares it against other baselines like **UCB**, **FedAvg**, **FedProx**, **S-FedAvg**, **Power-Of-Choice** and **Centralised** training on the **MNIST**, **FMINST**, and **CIFAR-10** datasets. Results are logged and visualized using W&B.
***

To run these algorithms execute ```main.py``` with the desired settings (edit the file).

Dataset Configuration:
1. name of dataset (from ```['fmnist','cifar10','mnist']```)
2. number of clients ($N$, any positive integer)
3. alpha ($\alpha$ parameter for dirichlet distribution, not required for Synthetic dataset) (typically varied in powers of 10 from $10^{-4}$ to $10^4$)

Algorithm Configuration:
1. algorithms to execute (from ```['greedyfed','fedavg','fedprox','sfedavg','ucb','centralised','poc']```)
2. client select fraction $\frac{M}{N}$
3. E, B, lr, momentum (epochs, batches, learning rate, SGD momentum)
4. T (number of communication rounds)
5. noise level (maximum client update noise in the privacy preserving setting)

Algorithm Hyperparameters
1. S-FedAvg [REF1] ( $\alpha = 1- \beta$)
2. Power-Of-Choice [REF2] (decay factor $\lambda$)
3. FedProx [REF3] (weight of proximal term $\mu$) 
4. GreedyFed (memory, weight for exponentially weighted average)

Logging results:
if logging is set to True the runs are saved on W&B

You can set the above parameters to a single value or implement a hyperparameter sweep over a list of values. After selecting the desired values, execute the following
```
python main.py
```

### plotting.py
To tabulate results, we download the runs from W&B into a Pandas DataFrame and calculate the test accuracy under various settings. The code directly produces the LaTeX code for tables used in our paper.

Set ```download = True``` in ```plotting.py``` if you wish to download results from your own runs instead of using existing logs. Set ```dataset``` to one of ```["mnist", "fmnist", "cifar10"]```
```
python plotting.py
```

### server.py
Implements the Server class with methods for:
1. client model aggregation
2. Shapley Value computation
3. returning server model performance metrics (accuracy and loss)

Three different kinds of Shapley Value estimation have been implemented in ```server.py```:
1. Truncated Monte Carlo (TMC) sampling 
2. GTG-Shapley (default) [REF4]
3. True Shapley Value [REF5] (extremely expensive to compute, computes loss over all subsets)

Convergence criterion for Shapley Values is implemented in ```utils.py```

### client.py
Implements the Client class with methods for:
1. client model training
2. adding noise to updates
3. returning client model performance metrics on client data (accuracy and loss)

### algorithms.py
Implements all the above-mentioned Federated Learning algorithms. Every method returns ```test_accuracy, train_accuracy, train_loss, validation_loss, test_loss, client_selections``` and some additional algorithm-specific metrics.
FedProx and FedAvg loss are defined using nested functions. The returned loss functions have a slightly different signature from those in PyTorch.

### data_preprocess.py
Implements methods for downloading and splitting datasets into train-val-test and splitting data across clients using the power law and Dirichlet distribution.

### initialise.py
Constructs server object and desired number of client objects with data and models allocated to each of them.

### model.py
Implements two different models: a Multi-Layer-Perceptron (NN) and a Convolutional Neural Network (CNN)

### utils.py
implements some utility functions

# Citations
[REF1] ...
[REF2]
# Citation
If you use this repo in your project or research, please cite as
***@software{singha23,
    author={Pranava Singhal, Shashi Raj Pandey and Petar Popovski},
    title={greedyfed},
    url={https://github.com/pringlesinghal/GreedyFed/},
    year={2023}
}
***




