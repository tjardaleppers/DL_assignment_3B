import EllmanNN 
import PytorchRNN
import LSTMNN

#######Q4
# Run Own implementation of Ellman
Q4 = False
if Q4:
    print('Training own implementation of Ellman')
    param = {'emb_size': 300,
            'hidden_size': 300,
            'lr': 1e-5}
    EllmanNN.trainModel(param)
    print()


#######Q5
# Tune hyperparameters for Elman implementation of Pytorch (RNN) and LSTM
from bayes_opt import BayesianOptimization

hyper_param_space = {
    'emb_size': (100, 500),
    'hidden_size': (100, 500),
    'lr': (1e-6, 1e-2),
}
RNN = False
LSTM = True

if RNN:
    print("nn.RRN")
    optimizer = BayesianOptimization(
        f=PytorchRNN.trainModel,
        pbounds=hyper_param_space,
        random_state=1
    )
    result = optimizer.maximize(init_points=1, n_iter=10)

if LSTM:
    print("nn.LSTM")
    optimizer = BayesianOptimization(
        f=LSTMNN.trainModel,
        pbounds=hyper_param_space,
        random_state=1
    )
    result = optimizer.maximize(init_points=1, n_iter=10)
