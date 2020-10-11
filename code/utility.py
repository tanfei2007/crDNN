#****************
# import modules
#***************
import numpy as np
import h5py
from keras import backend as K

#*******************************
# convert cumulative multiclass
# labels to  multiclass and 
# binary default/fully paid 
#******************************

def to_type_truth(data, threshold = 0.5):
	
	pred_prob = 1 - data[:,59]/np.array(data[:,0], dtype=np.float)
	pred_type = np.array(pred_prob > threshold, dtype=np.int)
	
	pred_type = pred_type.reshape((pred_type.shape[0], 1))
	pred_prob = pred_prob.reshape((pred_prob.shape[0], 1))	
	
	return pred_type, pred_prob

def to_surv_truth_1(data, threshold = 0.5):
	data = data[:, range(59) + [-1]]
	rslt = np.zeros((data.shape[0],))
	for i in range(data.shape[0]):
		idx = np.where(data[i, :] < threshold)
		idx = idx[0]
		if len(idx) == 0:
			rslt[i] = 59
		else:
			rslt[i] = idx[0] - 1
	return rslt


def to_surv_truth_0(data, threshold = 0.5):
        data = data[:, range(35) + [94]]
        rslt = np.zeros((data.shape[0],))
        for i in range(data.shape[0]):
                idx = np.where(data[i, :] < threshold)
                idx = idx[0]
                if len(idx) == 0:
                        rslt[i] = 35
                else:
                     	rslt[i] = idx[0] - 1
        return rslt


def to_surv_est_1(data, threshold = 0.5):
        x = data[:, range(59) + [-1]]
        del data
        rslt = np.zeros((x.shape[0],))
        for i in range(x.shape[0]):
                #rslt[i] = np.sum(x[i, :])
                idx = np.where(x[i, :] < threshold)
                idx = idx[0]
                if len(idx) == 0:
                        rslt[i] = 59
                else:
                        rslt[i] = idx[0] - 1
        return rslt


def to_surv_est_0(data, threshold = 0.5):
        x = data[:, range(35) + [94]]
        del data
        rslt = np.zeros((x.shape[0],))
        for i in range(x.shape[0]):
                #rslt[i] = np.sum(x[i, :])
                idx = np.where(x[i, :] < threshold)
                idx = idx[0]
                if len(idx) == 0:
                        rslt[i] = 35
                else:
                        rslt[i] = idx[0] - 1
        return rslt


def specifity_at_k(pred_class, pred_type, k = 10):
	idx = pred_class.argsort()[-k:]
	rslt = 1 - pred_type[idx].sum()/k
	return(rslt)

def load_data(file_path, file_type):	
	data_block = h5py.File(file_path + file_type + '.hdf5', 'r')
	print('loaded')
	X = np.array(data_block['X'])
	mask = np.array(data_block['mask'])
	y = np.array(data_block['y'])
	#y = y.reshape((y.shape[0], y.shape[1], 1))
	term = np.array(data_block['term'])
	term = term.reshape((term.shape[0], 1))
	survival = np.array(data_block['survival'])
	survival = survival.reshape((survival.shape[0], 1))
	data_block.close()
	print(y[0:2,])
	print('label:count', np.unique(y,  return_counts=True))
	return X, mask, y, term, survival


def weighted_binary_crossentropy(y_true, y_pred, weights):
	return K.mean(K.binary_crossentropy(y_true, y_pred) * weights, axis=-1)

#36 months 
def weighted_binary_crossentropy_0(y_true, y_pred, weights, terms):
        return K.mean(K.binary_crossentropy(y_true, y_pred) * weights * (1 - terms), axis=-1)

#60 months 
def weighted_binary_crossentropy_1(y_true, y_pred, weights, terms):
        return K.mean(K.binary_crossentropy(y_true, y_pred) * weights * terms, axis=-1)


#36 months known + censor
def weighted_binary_crossentropy_0_kc(y_true, y_pred, weights, terms, survivals):
        return K.mean(K.binary_crossentropy(y_true, y_pred) * weights * (1 - terms), axis=-1)

#36 months known
def weighted_binary_crossentropy_0_k(y_true, y_pred, weights, terms, survivals):
	#survivals[np.where(survivals != 0)] = 1
        return K.mean(K.binary_crossentropy(y_true, y_pred) * weights * (1 - terms) * survivals, axis=-1)

#60 months known + censor
def weighted_binary_crossentropy_1_kc(y_true, y_pred, weights, terms, survivals):
        return K.mean(K.binary_crossentropy(y_true, y_pred) * weights * terms, axis=-1)

#60 months known
def weighted_binary_crossentropy_1_k(y_true, y_pred, weights, terms, survivals):
	#survivals[np.where(survivals != 0)] = 1
        return K.mean(K.binary_crossentropy(y_true, y_pred) * weights * terms * survivals, axis=-1)



def cum2prob_partial_order(data, terms):
	prob = np.zeros_like(data)
	for i in range(data.shape[0]):
		x = data[i, :]
		term = terms[i,0]
		if term == 1:
			temp_end = x[59:-1] - x[60:]
			temp_start = x[:58] - x[1:59] 
			prob[i, 59:-1] = temp_end
			prob[i, -1] = x[-1]
			prob[i, 58] = x[58] - x[-2]
			prob[i, :58] = temp_start - temp_end[:-1]
		else:
			temp_end = x[59:94] - x[60:95]
                        temp_start = x[:34] - x[1:35]
                        prob[i, 59:94] = temp_end
                        prob[i, 94] = x[94]
                        prob[i, 34] = x[34] - x[93]
                        prob[i, :34] = temp_start - temp_end[:-1]
		
		idx_neg = np.where(prob[i, :] < 0)
		if len(idx_neg[0]) > 0:
			prob[i, idx_neg] = (1. - np.sum(prob[i, :]))/len(idx_neg[0]) #adjust prob for negative categories
		
	return prob
