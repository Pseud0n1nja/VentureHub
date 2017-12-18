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
    def __init__(self, R_indices, R, BATCH_SIZE):
        self.R_indices = R_indices
        self.R = R
        self.BATCH_SIZE = BATCH_SIZE
        self.i0 = np.inf
        self.i1 = np.inf
        self.epoch = 0
        self.broken = False
        self.last_batch = False

    def next(self):
        if self.i1 >= len(self.R):
            # new epoch.
            self.epoch += 1
            self.broken = False
            self.last_batch = False
            # reset the counter. 
            self.i0 = 0
            self.i1 = self.i0 + self.BATCH_SIZE
            print('Epoch %d %s' % (self.epoch, '_'*73))
        else:
            self.i0 = self.i0 + self.BATCH_SIZE
            self.i1 = min(self.i0 + self.BATCH_SIZE, len(self.R))
            if self.i1 - self.i0 < self.BATCH_SIZE:
                # broken batch.
                self.broken = True
            if self.i1 == len(self.R):
                self.last_batch = True
        #
        return self.R_indices[self.i0:self.i1], self.R[self.i0:self.i1]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def evaluate_preds_and_mae(sess, cv_R, cv_R_indices, R_pred, R_indices, R, BATCH_SIZE):
    i0 = (-1) * BATCH_SIZE
    i1 = 0
    preds = np.zeros((len(cv_R) // BATCH_SIZE) * BATCH_SIZE)
    for step in range(len(cv_R) // BATCH_SIZE):
        i0 = i0 + BATCH_SIZE
        i1 = i0 + BATCH_SIZE
        preds[i0:i1] = sess.run(R_pred, feed_dict={R_indices: cv_R_indices[i0:i1], R: cv_R[i0:i1]})

    mae = np.mean(np.abs(cv_R[:i1] - preds))
    
    return preds, mae

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def npy_read_data():
    print('Reading train_*.npy , cv_*.npy , test_*.npy files....')
    
    with open('data/train_R_indices.npy', 'rb') as f:
        train_R_indices = np.load(f)
    with open('data/train_R.npy', 'rb') as f:
        train_R = np.load(f)

    with open('data/cv_R_indices.npy', 'rb') as f:
        cv_R_indices = np.load(f)
    with open('data/cv_R.npy', 'rb') as f:
        cv_R = np.load(f)

    with open('data/test_R_indices.npy', 'rb') as f:
        test_R_indices = np.load(f)
    with open('data/test_R.npy', 'rb') as f:
        test_R = np.load(f)

    return train_R_indices, train_R, cv_R_indices, cv_R, test_R_indices, test_R

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def get_stacked_UV(R_indices, R, U, V, k, BATCH_SIZE):
    u_idx = R_indices[:,0]
    v_idx = R_indices[:,1]
    rows_U = tf.transpose(np.ones((k,1), dtype=np.int32)*u_idx)
    rows_V = tf.transpose(np.ones((k,1), dtype=np.int32)*v_idx)
    cols = np.arange(k, dtype=np.int32).reshape((1,-1))
    cols = tf.tile(cols, [BATCH_SIZE,1])

    indices_U = tf.stack([rows_U, cols], -1)
    indices_V = tf.stack([rows_V, cols], -1)
    stacked_U = tf.gather_nd(U, indices_U)
    stacked_V = tf.gather_nd(V, indices_V)
    # .....................................
        
    return stacked_U, stacked_V

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def construct_graph(LAMBDA=0, k=10, lr=0.001, BATCH_SIZE=256, n_investors=41838, n_startups=64764, cui=0.1):

    R_indices = tf.placeholder(dtype=tf.int32, shape=(BATCH_SIZE,2))
    R = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE,))
    
    # initialization of U and V is critical. 
    # set mean=np.sqrt(mu/k), where mu ~ 3 or 3.5
    U = tf.Variable(tf.truncated_normal(shape=(n_investors,k), mean=np.sqrt(3.5/k), stddev=0.2), dtype=tf.float32)
    V = tf.Variable(tf.truncated_normal(shape=(n_startups,k), mean=np.sqrt(3.5/k), stddev=0.2), dtype=tf.float32)

    # weights for cross-features
    X_UV = tf.Variable(tf.truncated_normal(shape=(k,k), mean=0, stddev=0.2), dtype=tf.float32)
    
    #.............................................. 
    
    stacked_U, stacked_V = get_stacked_UV(R_indices, R, U, V, k, BATCH_SIZE)

    # the term `tf.reduce_sum(U**2)` without passing an axis parameter sums up all the elements of matrix U**2.
    # Return value is a scalar.
    reg = (tf.reduce_sum((stacked_U)**2) + 
           tf.reduce_sum((stacked_V)**2) + 
           tf.reduce_sum((X_UV**2))) / (BATCH_SIZE*k)
    
    # the term `tf.multiply(stacked_U, stacked_V)` is elementwise multiplication.
    # Applying tf.reduce_sum(M, axis=1)--where M is a matrix--will sum all rows and return a column vector.
    # R_pred is a column vector of ratings corresponding to R_indices
    
    lin = tf.reduce_sum(tf.multiply(stacked_U, stacked_V), axis=1) 
    
    # ...........................................................
    
    xft = X_UV[0,0] * stacked_U[:,0] * stacked_V[:,0]
    for p in range(k):
        for q in range(k):
            xft += X_UV[p,q] * stacked_U[:,p] * stacked_V[:,q]
    # ...........................................................

    R_pred = tf.sigmoid(lin + xft)
    
    coeff = cui + (1 - cui) * R
    # loss: L2-norm of difference btw actual and predicted ratings
    loss = tf.sqrt(tf.reduce_sum(coeff * (R - R_pred)**2)/BATCH_SIZE) + LAMBDA * reg

    # Define train op.
    train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    
    return train, loss, reg, R_indices, R, U, V, R_pred, X_UV

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def train_the_model(R_indices, R, train_R_indices, train_R, BATCH_SIZE, 
               NUM_EPOCHS, LAMBDA, k, lr, 
               train, loss, reg, U, V, X_UV, R_pred,
               cv_R, cv_R_indices, test_R, test_R_indices):
    start = time.time()
    n_batches = len(train_R) // BATCH_SIZE
    init = tf.global_variables_initializer()
    batch = Batch(train_R_indices, train_R, BATCH_SIZE=BATCH_SIZE)
    _loss, _reg, batch_no = 0, 0, 0
    mae_train_arr, mae_cv_arr , mae_test_arr, loss_arr = [], [], [], []

    best_save_score = -np.inf
    
    if 'out.txt' in os.listdir(): 
        os.remove('out.txt')
    f_out = open('out.txt', 'a+') # append to file
    dp = 'NUM_EPOCHS: {}\nLAMBDA: {}\nk: {}\nlr: {}\nn_batches: {}\nBATCH_SIZE: {}'.format(\
                                            NUM_EPOCHS, LAMBDA, k, lr, n_batches, BATCH_SIZE)
    f_out.write(dp)
    print(dp)
    
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    
    print('start SGD iterations...')
    
    with tf.Session() as sess:
        sess.run(init)
        epoch_end = time.time()
        while not (batch.epoch == NUM_EPOCHS and batch.last_batch == True):
            batch_R_indices, batch_R = batch.next()
            if not batch.broken:
                batch_no += 1
                _, _batch_loss, _batch_reg = sess.run([train, loss, reg], 
                                        feed_dict={R_indices: batch_R_indices, R: batch_R})
                _loss += _batch_loss
                _reg += _batch_reg
                print("batch_no: {}, _loss estimate: {:6.4f}, t={:6.2f} sec".format(
                        batch_no, _loss/batch_no, time.time()-epoch_end), end='\r') 
            
            if batch.last_batch:  
                # fetch the mae's
                print('\nEvaluating MAE on training set...', end='\r')
                _, _mae_train = evaluate_preds_and_mae(sess, train_R, train_R_indices, R_pred, R_indices, R, BATCH_SIZE)
                print('Evaluating MAE on cv set...        ', end='\r')
                preds_cv, _mae_cv = evaluate_preds_and_mae(sess, cv_R, cv_R_indices, R_pred, R_indices, R, BATCH_SIZE)
                print('Evaluating MAE on test set...       ', end='\r')
                preds_test, _mae_test = evaluate_preds_and_mae(sess, test_R, test_R_indices, R_pred, R_indices, R, BATCH_SIZE)
                
                # threshold=0.7 <== an important parameter !!!
                print('Calculating ROC curve              ', end='\r')
                _, _, _precision_cv, _recall_cv, _f1_score_cv, _ = return_ROC_statistics(preds_cv, cv_R, threshold=.7)
                print('                                   ', end='\r')

                mae_train_arr.append(_mae_train)
                mae_cv_arr.append(_mae_cv)
                mae_test_arr.append(_mae_test)
                loss_arr.append(_loss/n_batches)
                mean_preds = np.mean(preds_cv)
                
                # printing....
                dp = '\nmae_train: %6.4f, **mae_cv: %6.4f**, mae_test: %6.4f,  mean(preds_cv): %6.4f' % \
                      (_mae_train, _mae_cv, _mae_test, np.mean(preds_cv))
                f_out.write(dp)
                print(dp, end = '')
                
                #dp = '(_reg/_loss) fraction: %6.4f' % (_reg/_loss)
                #f_out.write(dp)
                #print(dp)
                
                # resetting some iteration variables....
                batch_no, _loss, _reg = 0, 0, 0
                
                epoch_end = time.time()
                
                # Save model if recall_cv has reached a minimum.
                if (_f1_score_cv > best_save_score):
                    best_save_score = _f1_score_cv
                    save_path = saver.save(sess, "saved_models/best_model.ckpt")
                    print(',  CHECKPOINT!!', end='')
                
                print(', f1_score: {:6.4f}'.format(_f1_score_cv))
                
    f_out.close()
    return mae_train_arr, mae_cv_arr, mae_test_arr, loss_arr
    
    
    
    