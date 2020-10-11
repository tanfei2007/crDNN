#************************
# import modules
#************************
import os
#os.environ['THEANO_FLAGS'] = "device=gpu0"
import sys
import numpy as np
import h5py
from sklearn.metrics import roc_auc_score, matthews_corrcoef, precision_recall_fscore_support, average_precision_score, mean_squared_error
from pandas import DataFrame

from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.layers import Input
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D, AveragePooling1D
from keras.layers.pooling import AveragePooling1D
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l1,l2
from keras.layers.normalization import BatchNormalization
from utility import *
from keras.layers.advanced_activations import LeakyReLU, PReLU
#from letor_metrics import ndcg_score
from math import *
from functools import partial
from lifelines.utils import concordance_index

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.setrecursionlimit(15000)
np.random.seed(526) # for reproducibility



#setting
fold = sys.argv[1]
file_path = '../data/' + str(fold) + '/'
model_file = 'bestmodel_' + str(fold) + '.hdf5'

#******************
#load data
#******************
print('loading data')

X_train, mask_train, y_train, term_train, survival_train = load_data(file_path, 'train')
X_valid, mask_valid, y_valid, term_valid, survival_valid = load_data(file_path, 'valid')
X_test, mask_test, y_test, term_test, survival_test = load_data(file_path, 'test')


#***************
# load model
#**************
w_input = Input(shape=(y_train.shape[1],))
term_input = Input(shape=(1,))
survival_input = Input(shape=(1,))

w_loss_0_kc = partial(weighted_binary_crossentropy_0_kc, weights=w_input, terms=term_input, survivals=survival_input)
w_loss_0_kc.__name__ = 'w_loss_0_kc'

w_loss_0_k = partial(weighted_binary_crossentropy_0_k, weights=w_input, terms=term_input, survivals=survival_input)
w_loss_0_k.__name__ = 'w_loss_0_k'

w_loss_1_kc = partial(weighted_binary_crossentropy_1_kc, weights=w_input, terms=term_input, survivals=survival_input)
w_loss_1_kc.__name__ = 'w_loss_1_kc'

w_loss_1_k = partial(weighted_binary_crossentropy_1_k, weights=w_input, terms=term_input, survivals=survival_input)
w_loss_1_k.__name__ = 'w_loss_1_k'



model = load_model(model_file, custom_objects={'w_loss_0_kc': w_loss_0_kc,
                   'w_loss_0_k': w_loss_0_k, 'w_loss_1_kc': w_loss_1_kc, 'w_loss_1_k': w_loss_1_k})


#***********
# functions
#***********
def get_fork_prob(rslt_0, rslt_1, term):
	term = term.flatten()
	mask_0 = np.zeros_like(rslt_0)
	mask_1 = np.zeros_like(rslt_1)
	idx_0 = np.where(term == 0)
	idx_1 = np.where(term == 1)
	
	mask_0[idx_0] = [1]*35 + [0]*24 + [1]*36 + [0]*24
	mask_1[idx_1] =	[1]*119

	prob = rslt_0 * mask_0 + rslt_1 * mask_1

	return prob


def evaluate(X, mask, y, term, survival, file_name):
    #auc
    prob_0_kc, prob_0_k, prob_1_kc, prob_1_k = model.predict(X, verbose=1, batch_size=10000)
    predrslts_kc = get_fork_prob(prob_0_kc, prob_1_kc, term)
    predrslts_k = get_fork_prob(prob_0_k, prob_1_k, term)	
    

    #get different metrics
    pred_type, pred_prob = to_type_truth(predrslts_k)

    #loans with known statuses
    idx_k = np.where(survival != 0)
    y_type = survival[idx_k]
    y_type[y_type == 2] = 0
    print survival.shape
    print y_type.shape
    print pred_prob.shape
    print idx_k[0].shape
    print pred_prob[idx_k].shape

    auc = roc_auc_score(y_type, pred_prob[idx_k])
    aucpr = average_precision_score(y_type, pred_prob[idx_k], average="micro")
    mcc = matthews_corrcoef(y_type, pred_type[idx_k])
    prfs = precision_recall_fscore_support(y_type, pred_type[idx_k])

    print('label:count', np.unique(y_type,  return_counts=True))

  
    print 'negative ---> precision:%s, recall:%s, f1score:%s, support:%s' %(prfs[0][0], prfs[1][0], prfs[2][0], prfs[3][0])
    print 'positive ---> precision:%s, recall:%s, f1score:%s, support:%s' %(prfs[0][1], prfs[1][1], prfs[2][1], prfs[3][1])

    print('auc:', auc, 'aucpr:', aucpr, 'mcc:', mcc)
    
    #loans with both known and censored loans
    idx_0 = np.where(term[:, 0] == 0)
    idx_1 = np.where(term[:, 0] == 1)
    event_flag = survival.flatten()
    event_flag[event_flag !=0 ] = 1
    print predrslts_kc.shape
    print predrslts_kc[idx_1].shape	
    pred_surv_0 = to_surv_est_0(predrslts_kc[idx_0])
    pred_surv_1 = to_surv_est_1(predrslts_kc[idx_1])
    true_surv_0 = to_surv_truth_0(y[idx_0])
    true_surv_1 = to_surv_truth_1(y[idx_1])
    pred_surv = np.concatenate((pred_surv_0, pred_surv_1), axis = 0)
    true_surv = np.concatenate((true_surv_0, true_surv_1), axis = 0)
    c_index_all = concordance_index(true_surv, pred_surv, event_flag)
    c_index_0 = concordance_index(true_surv_0, pred_surv_0, event_flag[idx_0])
    c_index_1 = concordance_index(true_surv_1, pred_surv_1, event_flag[idx_1])


    print('c_index_all:', c_index_all, 'c_index_1:', c_index_1, 'c_index_0:', c_index_0)
    
    df_prob = DataFrame(pred_prob, index = range(pred_prob.shape[0]), columns = range(1))
    df_prob.to_csv(file_name + '_prob_' + str(fold) + '.csv', index = False)
    df_surv = DataFrame(pred_surv, index = range(pred_surv.shape[0]), columns = range(1))
    df_surv.to_csv(file_name + '_surv_' + str(fold) + '.csv', index = False)
	
    #calculate probs of all classes
    prob_class = cum2prob_partial_order(predrslts_k, term)

    df_prob_class = DataFrame(prob_class, index = range(prob_class.shape[0]), columns = range(prob_class.shape[1]))
    df_prob_class.to_csv(file_name + '_prob_class_' + str(fold) + '.csv', index = False)
    
    #df_kc = DataFrame(predrslts_kc, index = range(predrslts_kc.shape[0]), columns = range(predrslts_kc.shape[1]))
    #df_kc.to_csv(file_name + '_rslt_kc.csv', index = False)
    #df_k = DataFrame(predrslts_k, index = range(predrslts_k.shape[0]), columns = range(predrslts_k.shape[1]))
    #df_k.to_csv(file_name + '_rslt_k.csv', index = False)
    	

#**************
# on validation
# dataset
#**************
#print('predicting on valid sequences')
#evaluate(X_valid, mask_valid, y_valid, term_valid, survival_valid, 'valid')

print('predicting on test sequences')
evaluate(X_test, mask_test, y_test, term_test, survival_test, 'test')


