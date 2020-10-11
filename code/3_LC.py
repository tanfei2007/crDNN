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

survival_train[np.where(survival_train != 0)] = 1
survival_valid[np.where(survival_valid != 0)] =	1
survival_test[np.where(survival_test != 0)] =	1

#*************************
#building model
#*************************
print('building model')

main_input = Input(shape=(X_train.shape[1],), name='main_input')
w_input = Input(shape=(y_train.shape[1],))
term_input = Input(shape=(1,))
survival_input = Input(shape=(1,))

#*********
# FC 1
#*********

main_path = Dense(input_dim = X_train.shape[1], output_dim = 200, init = 'normal')(main_input)
main_path = LeakyReLU(alpha=.001)(main_path)
main_path = Dropout(0.5)(main_path)

#***********
# FC 2
#***********
main_path = Dense(output_dim = 200, init = 'normal')(main_path)
main_path = LeakyReLU(alpha=.001)(main_path)
main_path = Dropout(0.5)(main_path)

#***********
# FC 3
#***********
main_path = Dense(output_dim = 200, init = 'normal')(main_path)
main_path = LeakyReLU(alpha=.001)(main_path)
main_path = Dropout(0.4)(main_path)

#************
# FC 4
#***********
main_path = Dense(output_dim = 200, init = 'normal')(main_path)
main_path = LeakyReLU(alpha=.001)(main_path)
main_path = Dropout(0.4)(main_path)


#*****************
# Forked layer 0
# 36 months
#*****************
fork_path_0_kc = Dense(output_dim = 200, init = 'normal')(main_path)
fork_path_0_kc = Dense(output_dim = 200, init = 'normal')(fork_path_0_kc)
fork_path_0_kc = Dense(output_dim = y_train.shape[1], activation='sigmoid', name='0_kc_output')(fork_path_0_kc)

fork_path_0_k = Dense(output_dim = 200, init = 'normal')(main_path)
fork_path_0_k = Dense(output_dim = 200, init = 'normal')(fork_path_0_k)
fork_path_0_k = Dense(output_dim = y_train.shape[1], activation='sigmoid', name='0_k_output')(fork_path_0_k)

#*****************
# Forked layer 1
# 60 months
#*****************
fork_path_1_kc = Dense(output_dim = 200, init = 'normal')(main_path)
fork_path_1_kc = Dense(output_dim = 200, init = 'normal')(fork_path_1_kc)
fork_path_1_kc = Dense(output_dim = y_train.shape[1], activation='sigmoid', name='1_kc_output')(fork_path_1_kc)

fork_path_1_k = Dense(output_dim = 200, init = 'normal')(main_path)
fork_path_1_k = Dense(output_dim = 200, init = 'normal')(fork_path_1_k)
fork_path_1_k = Dense(output_dim = y_train.shape[1], activation='sigmoid', name='1_k_output')(fork_path_1_k)

#construct loss
w_loss_0_kc = partial(weighted_binary_crossentropy_0_kc, weights=w_input, terms=term_input, survivals=survival_input)
w_loss_0_kc.__name__ = 'w_loss_0_kc'

w_loss_0_k = partial(weighted_binary_crossentropy_0_k, weights=w_input, terms=term_input, survivals=survival_input)
w_loss_0_k.__name__ = 'w_loss_0_k'

w_loss_1_kc = partial(weighted_binary_crossentropy_1_kc, weights=w_input, terms=term_input, survivals=survival_input)
w_loss_1_kc.__name__ = 'w_loss_1_kc'

w_loss_1_k = partial(weighted_binary_crossentropy_1_k, weights=w_input, terms=term_input, survivals=survival_input)
w_loss_1_k.__name__ = 'w_loss_1_k'


model = Model([main_input, w_input, term_input, survival_input], \
	      [fork_path_0_kc, fork_path_0_k, fork_path_1_kc, fork_path_1_k])
#********************************************************


print('compiling model')
sgd = SGD(lr = 0.01, momentum = 0.9, decay = 1e-6, nesterov = True)
model.compile(loss = [w_loss_0_kc, w_loss_0_k, w_loss_1_kc, w_loss_1_k],
              loss_weights=[1., 1., 1., 1.], optimizer = sgd, metrics = ['accuracy'])


checkpointer = ModelCheckpoint(filepath=model_file, verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=20, verbose=0)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                  patience=5, min_lr=0.001)


class_weight = {}
for i in range(y_train.shape[1]):
	class_weight[i] = 1.0
	if i == 59:
		class_weight[i]	= 1.0
print(class_weight)

print(model.summary())
Hist = model.fit([X_train, (1-mask_train), term_train, survival_train], [y_train, y_train, y_train, y_train], 
		batch_size=128, nb_epoch=500, shuffle=True, verbose = 2, 
		validation_data=([X_valid, (1-mask_valid), term_valid, survival_valid], 
		[y_valid, y_valid, y_valid, y_valid]), 
		callbacks=[checkpointer,earlystopper], class_weight = class_weight)


#*********************
#plot both training
# and validation loss
#*********************
k = 10
loss = Hist.history['loss']
val_loss = Hist.history['val_loss']
epoch = range(1,len(loss)+1)
plt.plot(epoch[k:], loss[k:])
plt.plot(epoch[k:], val_loss[k:])
plt.legend(['train_loss', 'valid_loss'], loc = 'upper right')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('monitor.png')








