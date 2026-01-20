
from VBNN_class import VBNN
import numpy as np
import jax.numpy as jnp
from copy import deepcopy
from scipy.stats import norm as gaussian
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def nll(y_true, m, std):
    nll = - jnp.mean(gaussian.logpdf(y_true.reshape(-1), loc = m.reshape(-1), scale =std.reshape(-1)))
    return nll

# checking if y given from the model is very "different" from truth 
def empirical_coverage(y_true, y_model, std, n_st_devs=2):
    i = 0
    for m in range(len(y_true)):
        if y_true[m] > y_model[m] - n_st_devs*std[m] and y_true[m] < y_model[m] + n_st_devs*std[m]:
            i+=1
    return i/len(y_true)

def get_data(x, y, rst, test_size = 0.1):
    """splits the data and scales the inputs 

    Args:
        x: predictors, shape: (N, D_x)
        y: output, shape (N, D_y)
        rst: random seed
        test_size (float, optional): Defaults to 0.1.

    Returns:
        tuple: scaled predictors and outputs for training and testing (x_train, x_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test =  train_test_split(x, y, test_size=test_size, random_state=rst)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_scaled_tr = scaler.transform(X_train)
    X_scaled_pr = scaler.transform(X_test)
    return X_scaled_tr, X_scaled_pr, y_train, y_test


class ELBO_Choice(VBNN):
    """
    Description:
    Use when want to access actuall model and not just the prediction.

    Args:
        dict (dict): dictionary with the parameters needed for the VBNN class
    """
    def __init__(self, params):

        if isinstance(params, dict):
            for key, val in params.items():
                setattr(self, key, val)
        else:
            print('want dict')


    def model_choice(self, runs = 50,  epochs=50, rate= 1e-6, dm = True):
        """chooses the best model from a set of models based on the ELBO value after 5 epochs and trains the best model.

        Args:
            runs (int, optional): among how many models to choose. Defaults to 50.
            epochs (int, optional): max number of iterations for training. Defaults to 50.
            rate (float, optional): rate for ELBO when to stop the training. Defaults to 1e-6.
            dm (bool, optional): whether to do the EM step for tau. Defaults to True.

        Returns:
            VBNN: the best model
        """
        elbom = []
        model =VBNN(self.x, self.y, self.D_a, self.L, self.T, self.wb_mode, self.big_S)
        model.algorithm(5, dm)
        elbom = model.elbo_total[-1]
        for _ in range(runs):
            model_new = VBNN(self.x, self.y, self.D_a, self.L, self.T, self.wb_mode, self.big_S)
            model_new.algorithm(5, dm)
            if model_new.elbo_total[-1] > elbom:
                model = deepcopy(model_new)
                elbom = np.copy(model.elbo_total[-1])
                
        model.algorithm(epochs = epochs, rate = rate, EM_step = dm)
        print('done training epochs: ', len(model.elbo_total))
        print('elbo: ', model.elbo_total[-1])
        return deepcopy(model)

class ELBO_Simple(VBNN):
    """
    Description:
    Use when don't want to access actuall model and want the prediction of one model with the best ELBO.

    Args:
        dict (dict): dictionary with the parameters needed for the VBNN class
    """
    

    def __init__(self, params):

        if isinstance(params, dict):
            for key, val in params.items():
                setattr(self, key, val)
        else:
            print('want dict')


    def model_choice(self, x_new, runs = 10,  epochs=50, epochs_pred=20, rate = 1e-5, rate_pred = 1e-4):
        """chooses the best model from a set of models based on the ELBO value after 5 epochs, trains the best model and gets the predictive mean and st dev.

        Args:
            x_new (np.array): new data to predict on
            runs (int, optional): among how many models to choose. Defaults to 10.
            epochs (int, optional): max number of iterations for training. Defaults to 50.
            epochs_pred (int, optional): max number of iterations for prediction. Defaults to 20.
            rate (float, optional): rate for ELBO when to stop the training. Defaults to 1e-6.
            rate_pred (float, optional): rate for ELBO when to stop the prediction. Defaults to 1e-4.
            
        """
        model =VBNN(self.x, self.y, self.D_a, self.L, self.T, self.wb_mode, self.big_S)
        model.algorithm(5)
        elbo_current = model.elbo_total[-1]
        for _ in range(runs):
            model_new = VBNN(self.x, self.y, self.D_a, self.L, self.T, self.wb_mode, self.big_S)
    
            model_new.algorithm(5)
            if model_new.elbo_total[-1] > elbo_current:
                model = deepcopy(model_new)
                elbo_current = np.copy(model.elbo_total[-1])

        self.prediction = None
        self.prediction_std = None
    
        model.algorithm(epochs, rate)
        print('done training epochs: ', len(model.elbo_total))
        print('elbo: ', model.elbo_total[-1])
        model.predict(np.copy(x_new), epochs_pred, rate_pred)
        print('done epochs for pred: ', len(model.elbo_pred))
        print('elbo for pred: ', model.elbo_pred[-1].item())
        self.prediction = np.copy(model.prediction_mean)
        self.prediction_std = np.sqrt(model.var_tot)



class ELBO_Mix(VBNN):
    """
    Description:
    Use when don't want to access actuall model and want the prediction of ensembles of models, two initialized with 'laplace' mode, two with 'spikeslab'.

    Args:
        dict (dict): dictionary with the parameters needed for the VBNN class, EXCEPT for wb_mode
    """
    def __init__(self, params):

        if isinstance(params, dict):
            for key, val in params.items():
                setattr(self, key, val)
        else:
            print('want dict')


    def model_ensemble(self, x_new, epochs=50, epochs_pred=20,  rate = 1e-5, rate_pred = 1e-4, tau=1):
        """trains 4 models and gets the predictive mean and st dev.

        Args:
            x_new (np.array): new data to predict on
            epochs (int, optional): max number of iterations for training. Defaults to 50.
            epochs_pred (int, optional): max number of iterations for prediction. Defaults to 20.
            rate (float, optional): rate for ELBO when to stop the training. Defaults to 1e-6.
            rate_pred (float, optional): rate for ELBO when to stop the prediction. Defaults to 1e-4.
            tau (int, optional): if you want to change the way models are combined (weights become proportional to exp(tau*elbo). Defaults to 1.
        """

        self.prediction = None
        self.weights = None
        self.prediction_std = None
        self.Ds = None
        self.y_model = []
        self.model_vartot = []

        self.elbos = []
        model_1 =VBNN(self.x, self.y, self.D_a, self.L, self.T, wb_mode = 'spikeslab', big_S = self.big_S)
        model_1.algorithm(epochs, rate)
        self.elbos.append(model_1.elbo_total[-1])
        model_1.predict(np.copy(x_new), epochs_pred, rate_pred)
        self.y_model.append(np.copy(model_1.prediction_mean))
        self.model_vartot.append(np.copy(model_1.var_tot))

        model_2 =VBNN(self.x, self.y, self.D_a, self.L, self.T, wb_mode = 'spikeslab', big_S = self.big_S)
        model_2.algorithm(epochs, rate)
        self.elbos.append(model_2.elbo_total[-1])
        model_2.predict(np.copy(x_new), epochs_pred, rate_pred)
        self.y_model.append(np.copy(model_2.prediction_mean))
        self.model_vartot.append(np.copy(model_2.var_tot))

        model_3 =VBNN(self.x, self.y, self.D_a, self.L, self.T, wb_mode = 'laplace', big_S = self.big_S)
        model_3.algorithm(epochs,rate)
        self.elbos.append(model_3.elbo_total[-1])
        model_3.predict(np.copy(x_new), epochs_pred, rate_pred)
        self.y_model.append(np.copy(model_3.prediction_mean))
        self.model_vartot.append(np.copy(model_3.var_tot))

        model_4 =VBNN(self.x, self.y, self.D_a, self.L, self.T, wb_mode = 'laplace', big_S = self.big_S)
        model_4.algorithm(epochs, rate)
        self.elbos.append(model_4.elbo_total[-1])
        model_4.predict(np.copy(x_new), epochs_pred, rate_pred)
        self.y_model.append(np.copy(model_4.prediction_mean))
        self.model_vartot.append(np.copy(model_4.var_tot))

        self.Ds = np.copy(model_3.D)
        self.weights = np.exp(tau*np.array(self.elbos) - tau*np.array(self.elbos).max()*np.ones(4))/sum(np.exp(tau*np.array(self.elbos) - tau*np.array(self.elbos).max()*np.ones(4)))
        self.prediction = np.sum([y*w for (y,w) in zip(self.y_model, self.weights)], axis=-3)
        self.prediction_std =  np.sqrt(np.sum([(vt*w).reshape(self.y_model[0].shape) + (y**2*w).reshape(self.y_model[0].shape) for (y,w, vt) in zip(self.y_model, self.weights, self.model_vartot)], axis=-3) - self.prediction**2)