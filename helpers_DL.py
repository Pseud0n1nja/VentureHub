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

def return_ROC_statistics(y_pred, y_true, threshold=np.linspace(0,1,26)):
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

def plotter(mae_train_arr, mae_cv_arr, mae_test_arr, loss_arr, BATCH_SIZE):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    n_epoch = len(mae_train_arr)
    ax1.plot(range(1, n_epoch+1), mae_test_arr, 'bd-', alpha=.6);
    ax1.plot(range(1, n_epoch+1), mae_cv_arr, 'rd-', alpha=.6);
    ax1.plot(range(1, n_epoch+1), mae_train_arr, 'kd-', alpha=.6);
    ax1.set_xlim(left=1);
    #ax1.set_ylim(0.5, 0.65)
    i_min = np.argmin(mae_cv_arr)
    ax1.set_title('MAE_cv: {:6.4f}, BATCH_SIZE: {}*256'.format(mae_cv_arr[i_min], BATCH_SIZE//256))
    ax1.legend(['test', 'cv', 'train'])
    ax2.plot(range(1, n_epoch+1), loss_arr, 'kd-', alpha=.6);
    
    ax2.set_title('Training Loss (RMS)')
    ax1.set_xlabel('epochs')
    ax2.set_xlabel('epochs')
    #ax2.set_ylim(0.7, 1)
    i_min = np.argmin(mae_cv_arr)
    ax1.plot(i_min+1,mae_cv_arr[i_min], 'ro', markersize=13,
                markeredgewidth=2, markerfacecolor='None')
    
    return ax1, ax2

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
        self.epoch = 0
        self.broken = False
        self.last_batch = False
        self.n_investors = n_investors
        self.n_input = n_input

    def next(self):
        if self.i1 >= len(self.R):
            # new epoch.
            self.epoch += 1
            self.broken = False
            self.last_batch = False
            # reset the counter. 
            self.i0 = 0
            self.i1 = self.i0 + self.BATCH_SIZE
            #print('Epoch %d %s' % (self.epoch, '_'*73))
        else:
            self.i0 = self.i0 + self.BATCH_SIZE
            self.i1 = min(self.i0 + self.BATCH_SIZE, len(self.R))
            if self.i1 == len(self.R):
                self.last_batch = True
            if self.i1 - self.i0 < self.BATCH_SIZE:
                self.broken = True
                return None, None
            

        X = np.zeros((self.BATCH_SIZE, self.n_input))
        X[np.arange(self.BATCH_SIZE), self.R_indices[self.i0:self.i1, 0]] = 1
        X[np.arange(self.BATCH_SIZE), self.n_investors + self.R_indices[self.i0:self.i1, 1]] = 1
        y = self.R[self.i0:self.i1].reshape(self.BATCH_SIZE,1)

        # add number of affiliations between investor and startup
        X[np.arange(self.BATCH_SIZE), -1*np.ones(self.BATCH_SIZE).astype(np.int64)] = 0
        
        return X, y

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def evaluate_preds_and_mae(sess, cv_R, cv_R_indices, n_investors, n_input, y_pred, X, y, BATCH_SIZE):     
    batch = Batch(cv_R_indices, cv_R, n_investors, n_input, BATCH_SIZE=BATCH_SIZE) 
    preds = np.zeros((len(cv_R) // BATCH_SIZE) * BATCH_SIZE)
    i0 = (-1) * BATCH_SIZE
    i1 = 0
    for step in range(len(cv_R) // BATCH_SIZE):
        batch_X, batch_y = batch.next()
        i0 = i0 + BATCH_SIZE
        i1 = i0 + BATCH_SIZE
        preds[i0:i1] = sess.run(y_pred, feed_dict={X: batch_X, y: batch_y})[:,0]
        
    mae = np.mean(np.abs(cv_R[:i1] - preds))
    
    return preds, mae

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def construct_graph(LAMBDA=0, lr=0.001, BATCH_SIZE=256, n_input=106603):
    
    X = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, n_input))
    y = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, 1))
    
    # Define layer weights as tf.Variable
    n_hidden_1, n_hidden_2 = 100, 100 
    
    
    weights = {
        'h1': tf.Variable(tf.truncated_normal(shape=(n_input, n_hidden_1), mean=np.sqrt(0), stddev=0.2), dtype=tf.float32),
        'h2': tf.Variable(tf.truncated_normal(shape=(n_hidden_1, n_hidden_2), mean=np.sqrt(0), stddev=0.2), dtype=tf.float32),
        'out': tf.Variable(tf.truncated_normal(shape=(n_hidden_2, 1), mean=np.sqrt(0), stddev=0.2), dtype=tf.float32) }
    
    biases = {
        'b1': tf.Variable(tf.truncated_normal(shape=(n_hidden_1,), mean=np.sqrt(0), stddev=0.2), dtype=tf.float32),
        'b2': tf.Variable(tf.truncated_normal(shape=(n_hidden_2,), mean=np.sqrt(0), stddev=0.2), dtype=tf.float32),
        'out': tf.Variable(tf.truncated_normal(shape=(1,), mean=np.sqrt(0), stddev=0.2), dtype=tf.float32) }
    # ..............................................................................................................
    # Add Hidden Layer 1
    layer_1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'])
    layer_1 = tf.tanh(layer_1)

    # Add Hidden Layer n
    layer_n = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_n = tf.tanh(layer_n)

    # Output Layer
    logits = tf.matmul(layer_n, weights['out']) + biases['out']    
    y_pred = tf.sigmoid(logits)
    # Define loss and optimizer
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train = optimizer.minimize(loss)

    return train, loss, X, y, y_pred

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def train_the_model(train_R_indices, train_R, cv_R_indices, cv_R, test_R_indices, test_R,
               BATCH_SIZE, NUM_EPOCHS, LAMBDA, lr, 
               train, loss, X, y, y_pred,
               n_investors, n_input):
    
    start = time.time()
    n_batches = len(train_R) // BATCH_SIZE
    init = tf.global_variables_initializer()
    batch = Batch(train_R_indices, train_R, n_investors, n_input, BATCH_SIZE=BATCH_SIZE)
    _loss, _reg, batch_no = 0, 0, 0
    mae_train_arr, mae_cv_arr , mae_test_arr, loss_arr = [], [], [], []
    
    best_save_score = -np.inf
    
    print('NUM_EPOCHS: {}\nLAMBDA: {}\nlr: {}\nn_batches: {}\nBATCH_SIZE: {}'.format(\
                                            NUM_EPOCHS, LAMBDA, lr, n_batches, BATCH_SIZE))
    
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(init)
        epoch_end = time.time()
        while not (batch.epoch == NUM_EPOCHS and batch.last_batch == True):
            batch_X, batch_y = batch.next()
            if batch.i0 == 0:
                print('Epoch %d %s' % (batch.epoch, '_'*73))
                
            if not batch.broken:
                batch_no += 1
                _, _batch_loss = sess.run([train, loss], feed_dict={X: batch_X, y: batch_y})
                _loss += _batch_loss
                
                print("batch_no: {}, _loss estimate: {:6.4f}, t={:6.1f} sec".format(
                        batch_no, _loss/batch_no, time.time()-epoch_end), end='\r') 
            
            if batch.last_batch:  
                # fetch the mae's
                print('\nEvaluating MAE on training set...', end='\r')
                ev_start = time.time()
                preds_train, _mae_train = evaluate_preds_and_mae(
                                    sess, train_R, train_R_indices, n_investors, n_input, y_pred, X, y, BATCH_SIZE)
                print('Evaluating MAE on cv set...      ', end='\r')
                preds_cv, _mae_cv = evaluate_preds_and_mae(
                                    sess, cv_R, cv_R_indices, n_investors, n_input, y_pred, X, y, BATCH_SIZE)
                print('Evaluating MAE on test set...    ', end='\r')
                preds_test, _mae_test = evaluate_preds_and_mae(
                                    sess, test_R, test_R_indices, n_investors, n_input, y_pred, X, y, BATCH_SIZE)
                
                # threshold=0.7 <== an important parameter !!!
                _, _, _precision_cv, _recall_cv, _f1_score_cv, _ = return_ROC_statistics(preds_cv, cv_R, threshold=.7)
                
                mae_train_arr.append(_mae_train)
                mae_cv_arr.append(_mae_cv)
                mae_test_arr.append(_mae_test)
                loss_arr.append(_loss/n_batches)
                mean_preds = np.mean(preds_cv)
                
                # printing....
                print('\nmae_train: %6.4f, **mae_cv: %6.4f**, mae_test: %6.4f,  mean(preds_cv): %6.4f, ' % \
                      (_mae_train, _mae_cv, _mae_test, np.mean(preds_cv)), end = '')
                
                # resetting some iteration variables....
                _loss_S = _loss
                batch_no, _loss, _reg = 0, 0, 0
                
                epoch_end = time.time()
                
                # Save model if recall_cv has reached a minimum.
                if (_f1_score_cv > best_save_score):
                    best_save_score = _f1_score_cv
                    save_path = saver.save(sess, "saved_models/best_model.ckpt")
                    print(',  CHECKPOINT!!', end='')
                
                print(', f1_score: {:6.4f}'.format(_f1_score_cv))
                
    print()
    return mae_train_arr, mae_cv_arr, mae_test_arr, loss_arr
    
    
    
    