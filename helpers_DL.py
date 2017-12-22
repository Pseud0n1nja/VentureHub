import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import time
import os
import matplotlib.pylab as pylab
from matplotlib.ticker import MaxNLocator

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def ROC_statistics(y_pred, y_true, threshold=np.linspace(0,1,26)):
    recall, f1_score, fpr, accuracy, precision = [], [], [], [], []
    
    if not hasattr(threshold, '__iter__'):
        threshold = [threshold]
        
    for t in threshold:
        y_pred_01  = threshold_rounder(y_pred, threshold=t)
        
        acc_, pre_, rec_, f1_score_, fpr_ = confusion_matrix(y_true[:len(y_pred)], y_pred_01)
        
        accuracy.append(acc_)
        precision.append(pre_)
        recall.append(rec_)
        f1_score.append(f1_score_)
        fpr.append(fpr_)
        
    if len(threshold) == 1:
        return y_pred_01, acc_, pre_, rec_, f1_score_, fpr_
    else:
        return y_pred_01, accuracy, precision, recall, f1_score, fpr

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def threshold_rounder(y_pred_test, threshold=0.5):
    y_pred_test_01 = np.zeros_like(y_pred_test)
    y_pred_test_01[y_pred_test >= threshold] = 1
    y_pred_test_01[y_pred_test < threshold] = 0

    return y_pred_test_01

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def confusion_matrix(y_true, y_pred):
    y_true = y_true[:len(y_pred)]
    
    a = y_pred[y_true == 1]
    b = y_pred[y_true == 0]
    
    tp = sum(a)
    fp = sum(b)
    tn = len(b) - sum(b)
    fn = len(a) - sum(a)
    
    accuracy  = (tp+tn)/float(tp+tn+fp+fn)
    recall    = tp/float(tp+fn)
    precision = tp/float(tp+fp)
    f1_score  = 2*precision*recall/float(precision + recall)
    fpr = fp/(fp+tn)
    
    return accuracy, precision, recall, f1_score, fpr

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def print_runtime(start, p_flag=True):
    end = time.time()
    if p_flag:
        print('Runtime: %d min %d sec' % ((end-start)//60, (end-start)%60))
        return None
    else:
        return 'Runtime: %d min %d sec' % ((end-start)//60, (end-start)%60)
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def plotter(loss_cv_arr, loss_train_arr, batch):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    n_epoch = len(loss_cv_arr)
    ax.plot(range(1, n_epoch+1), loss_cv_arr, 'bd-', alpha=.6);
    ax.plot(range(1, n_epoch+1), loss_train_arr, 'kd-', alpha=.6);
    
    i_min = np.argmin(loss_cv_arr)
    ax.set_title('loss_cv: {:6.4f}, BATCH_SIZE: {}*256'.format(loss_cv_arr[i_min], batch.BATCH_SIZE//256))
    ax.set_xlim(left=1)
    ax.set_xlabel('epochs')
    ax.plot(i_min+1, loss_cv_arr[i_min], 'b.', markersize=14,
                markeredgewidth=4, alpha=.75)
    
    i_min_f1 = batch.ckpt_epoch-1
    ax.plot(i_min_f1+1, loss_cv_arr[i_min_f1], 'ro', markersize=15,
                markeredgewidth=5, markerfacecolor='None', alpha=.75)

    ax.legend(['cv', 'train', 'argmin cv', 'argmin f1_score'])
    
    return ax

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def split_data(R_indices, R):
    # training data takes up 64% of shuffled data
    # cv data 16% and
    # test data 20%
    
    N = R.shape[0]
    i0 = int(N*.64)
    i1 = int(N*.80)
    
    train_R_indices = R_indices[:i0]
    train_R = R[:i0]
    
    cv_R_indices = R_indices[i0:i1]
    cv_R = R[i0:i1]
    
    test_R_indices = R_indices[i1:]
    test_R = R[i1:]
    
    return train_R_indices, train_R, cv_R_indices, cv_R, test_R_indices, test_R

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def shuffler(a,b):
    i = np.arange(a.shape[0])
    np.random.shuffle(i)
    a = a[i]
    b = b[i]
    return a,b
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

class Batch(object):
    def __init__(self, R_indices, R, n_investors, n_input, BATCH_SIZE):
        self.R_indices = R_indices
        self.R = R
        self.BATCH_SIZE = BATCH_SIZE
        self.i0 = np.inf
        self.i1 = np.inf
        self.batch_no = 0
        self.epoch = 0
        self.broken = False
        self.last_batch = False
        self.n_investors = n_investors
        self.n_input = n_input

    def next(self):
        if self.i1 >= len(self.R):
            # new epoch.
            self.epoch += 1
            self.batch_no = 1
            self.broken = False
            self.last_batch = False
            
            # reset the counter. 
            self.i0 = 0
            self.i1 = self.i0 + self.BATCH_SIZE
        else:
            self.batch_no += 1
            self.i0 = self.i0 + self.BATCH_SIZE
            self.i1 = min(self.i0 + self.BATCH_SIZE, len(self.R))
            if self.i1 == len(self.R):
                self.last_batch = True
            if self.i1 - self.i0 < self.BATCH_SIZE:
                self.broken = True
                # broken_batch_size
                bbs = self.i1-self.i0
                X = np.zeros((bbs, self.n_input))
                X[np.arange(bbs), self.R_indices[self.i0:self.i1, 0]] = 1
                X[np.arange(bbs), self.n_investors + self.R_indices[self.i0:self.i1, 1]] = 1
                y = self.R[self.i0:self.i1].reshape(bbs,1)
                return X, y
            
        # Create a zeros input vector
        X = np.zeros((self.BATCH_SIZE, self.n_input))
        # set the VC firm index to 1
        X[np.arange(self.BATCH_SIZE), self.R_indices[self.i0:self.i1, 0]] = 1
        # set the startup index to 1
        X[np.arange(self.BATCH_SIZE), self.n_investors + self.R_indices[self.i0:self.i1, 1]] = 1
        y = self.R[self.i0:self.i1].reshape(self.BATCH_SIZE,1)

        # add number of affiliations between investor and startup
        X[np.arange(self.BATCH_SIZE), (-1)*np.ones(self.BATCH_SIZE).astype(np.int64)] = 0
        
        return X, y

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def evaluate_preds_and_loss(sess, cv_R, cv_R_indices, loss_op, y_pred_op, X, y, BATCH_SIZE, n_investors, n_input):     
    preds = np.zeros(len(cv_R))
    epoch_loss_cv = 0
    batch = Batch(cv_R_indices, cv_R, n_investors, n_input, BATCH_SIZE=BATCH_SIZE) 

    while not batch.last_batch == True:
        batch_X, batch_y = batch.next()
        out = sess.run([loss_op, y_pred_op], feed_dict={X: batch_X, y: batch_y})
        batch_loss_cv, preds[batch.i0:batch.i1] = out[0], out[1][:,0]
        epoch_loss_cv += batch_loss_cv*(batch.i1-batch.i0)
        
    return preds, epoch_loss_cv/len(cv_R)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def construct_graph(LAMBDA=0, lr=0.001, BATCH_SIZE=256, n_input=106603):
    
    X = tf.placeholder(dtype=tf.float32, shape=(None, n_input))
    y = tf.placeholder(dtype=tf.float32, shape=(None, 1))
    
    n_hidden_1, n_hidden_2 = 100, 100 
    
    # Declare layer weights as tf.Variable
    weights = {
        'h1': tf.Variable(tf.truncated_normal(shape=(n_input, n_hidden_1), mean=np.sqrt(0), stddev=0.2), dtype=tf.float32),
        'h2': tf.Variable(tf.truncated_normal(shape=(n_hidden_1, n_hidden_2), mean=np.sqrt(0), stddev=0.2), dtype=tf.float32),
        'out': tf.Variable(tf.truncated_normal(shape=(n_hidden_2, 1), mean=np.sqrt(0), stddev=0.2), dtype=tf.float32) }
    
    biases = {
        'b1': tf.Variable(tf.truncated_normal(shape=(n_hidden_1,), mean=np.sqrt(0), stddev=0.2), dtype=tf.float32),
        'b2': tf.Variable(tf.truncated_normal(shape=(n_hidden_2,), mean=np.sqrt(0), stddev=0.2), dtype=tf.float32),
        'out': tf.Variable(tf.truncated_normal(shape=(1,), mean=np.sqrt(0), stddev=0.2), dtype=tf.float32) }
    # ..............................................................................................................
    # Hidden Layer 1
    layer_1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'])
    layer_1 = tf.tanh(layer_1)

    # Hidden Layer n
    layer_n = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_n = tf.tanh(layer_n)

    # Output Layer
    logit = tf.matmul(layer_n, weights['out']) + biases['out']    
    y_pred = tf.sigmoid(logit)
    # Define loss and optimizer
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train = optimizer.minimize(loss)

    return train, loss, X, y, y_pred

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def train_the_model(train_R_indices, train_R, cv_R_indices, cv_R, test_R_indices, test_R,
               BATCH_SIZE, NUM_EPOCHS, LAMBDA, lr, 
               train_op, loss_op, y_pred_op, X, y, 
               n_investors, n_input, threshold):
    
    start = time.time()
    n_batches = len(train_R) // BATCH_SIZE
    init = tf.global_variables_initializer()
    batch = Batch(train_R_indices, train_R, n_investors, n_input, BATCH_SIZE=BATCH_SIZE)
    epoch_loss_train, _reg = 0, 0
    loss_cv_arr , loss_train_arr = [], []
    
    best_save_score = -np.inf
    
    print('NUM_EPOCHS: {}\nLAMBDA: {}\nlr: {}\nn_batches: {}\nBATCH_SIZE: {}\nthreshold: {}'.format(\
                                            NUM_EPOCHS, LAMBDA, lr, n_batches, BATCH_SIZE, threshold))
    
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    print('start SGD iterations...', end='\r')
    
    with tf.Session() as sess:
        sess.run(init)
        epoch_end = time.time()
        while not (batch.epoch == NUM_EPOCHS and batch.last_batch == True):
            batch_X, batch_y = batch.next()
            if batch.i0 == 0:
                print('Epoch %d %s' % (batch.epoch, '_'*62))
                
            _, _batch_loss_train = sess.run([train_op, loss_op], feed_dict={X: batch_X, y: batch_y})

            epoch_loss_train += _batch_loss_train*(batch.i1-batch.i0)
            
            print("batch_no:{}/{}, loss_train:{:6.4f}, t={:0.1f} sec".format(
                                                    batch.batch_no, n_batches, epoch_loss_train/batch.i1,
                                                    time.time()-epoch_end), end='\r') 
            
            if batch.last_batch:  
                # collect some statistics for printing the loss function etc
                # fetch the losses
                
                print('\nEvaluating loss_cv and preds_cv on cv set... ', end='\r')
                preds_cv, _loss_cv = evaluate_preds_and_loss(
                                          sess, cv_R, cv_R_indices, loss_op, y_pred_op, X, y, BATCH_SIZE, n_investors, n_input)
                
                # threshold ~ 0.7 <== an important parameter !!!
                print('Calculating ROC curve, threshold: {:3.1f}         '.format(threshold), end='\r')
                #_, _, _precision_cv, _recall_cv, _f1_score_cv, _ = ROC_statistics(preds_cv, cv_R, threshold=threshold)
                
                _, _, _, _, f1_score_cv, _ = ROC_statistics(preds_cv, cv_R)
                # f1_score_cv is an array.
                idx = np.nanargmax(f1_score_cv)
                _f1_score_cv = f1_score_cv[idx]
                # retrieve the threshold value where f1_score_cv reaches a maximum.
                threshold = np.linspace(0,1,len(f1_score_cv))[idx]
                
                loss_cv_arr.append(_loss_cv)
                loss_train_arr.append(epoch_loss_train/(batch.i1))
                                
                # resetting some iteration variables....
                epoch_loss_train, _reg = 0, 0
                epoch_end = time.time()

                # Save model if _f1_score_cv has reached a minimum.
                if (_f1_score_cv > best_save_score):
                    best_save_score = _f1_score_cv
                    save_path = saver.save(sess, 'saved_models/DL_models/best_model.ckpt')
                    ckpt = '  !! CHECKPOINT!!'
                    batch.ckpt_epoch = batch.epoch
                else: 
                    ckpt = ''
                # printing....
                print('loss_train: {0:6.4f}, **loss_cv: {1:6.4f}**, f1_score_cv: {2:6.4f} @threshold:{3:1.2f} {4:3s}'.format(
                                                         loss_train_arr[-1], loss_cv_arr[-1], _f1_score_cv, threshold, ckpt))
                
                with open('saved_models/loss_train_and_cv.pkl','wb') as f:
                    pickle.dump((loss_train_arr, loss_cv_arr, batch), f)
                
    print()
    return loss_cv_arr, loss_train_arr, batch







