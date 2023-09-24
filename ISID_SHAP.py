# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os, itertools, random, argparse, time, datetime
import numpy as np
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score,explained_variance_score
from math import sqrt

import scipy.sparse as sp
from scipy.stats.stats import pearsonr
from data import *
from ISID_model import *
from ISID_wo_model import *


import shutil
import logging
import glob
import time
from tensorboardX import SummaryWriter
return_matrix = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp

# Training settings
ap = argparse.ArgumentParser()
ap.add_argument('--dataset', type=str, default='region785', help="Dataset string")
ap.add_argument('--sim_mat', type=str, default='region-adj', help="adjacency matrix filename (*-adj.txt)")
ap.add_argument('--n_layer', type=int, default=1, help="number of layers (default 1)") 
ap.add_argument('--n_hidden', type=int, default=20, help="rnn hidden states (could be set as any value)") 
ap.add_argument('--seed', type=int, default=42, help='random seed')
ap.add_argument('--epochs', type=int, default=100, help='1500 default number of epochs to train')
ap.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
ap.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
ap.add_argument('--dropout', type=float, default=0.2, help='dropout rate usually 0.2-0.5.')
ap.add_argument('--batch', type=int, default=32, help="batch size")
ap.add_argument('--check_point', type=int, default=1, help="check point")
ap.add_argument('--shuffle', action='store_true', default=False, help="not used, default false")
ap.add_argument('--train', type=float, default=.6, help="Training ratio (0, 1)")
ap.add_argument('--val', type=float, default=.2, help="Validation ratio (0, 1)")
ap.add_argument('--test', type=float, default=.2, help="Testing ratio (0, 1)")
ap.add_argument('--model', default='ISID', choices=['cola_gnn','CNNRNN_Res','RNN','AR','ARMA','VAR','GAR','SelfAttnRNN','lstnet','stgcn','dcrnn'], help='')
ap.add_argument('--rnn_model', default='RNN', choices=['LSTM','RNN','GRU'], help='')
ap.add_argument('--mylog', action='store_false', default=True,  help='save tensorboad log')
ap.add_argument('--cuda', action='store_true', default=True,  help='')
ap.add_argument('--window', type=int, default=20, help='')
ap.add_argument('--horizon', type=int, default=1, help='leadtime default 1')
ap.add_argument('--save_dir', type=str,  default='./src/save',help='dir path to save the final model')
ap.add_argument('--gpu', type=int, default=1,  help='choose gpu 0-10')
ap.add_argument('--lamda', type=float, default=0.01,  help='regularize params similarities of states')
ap.add_argument('--bi', action='store_true', default=False,  help='bidirectional default false')
ap.add_argument('--patience', type=int, default=200, help='patience default 100')
ap.add_argument('--k', type=int, default=10,  help='kernels')
ap.add_argument('--hidsp', type=int, default=15,  help='spatial dim')

ap.add_argument('--SID_D_dim', type=str, default=32, help='STID_D_dim')
ap.add_argument('--SID_layer_num', type=str, default=12, help='STID_layer_num')
ap.add_argument('--SID_emb_dim', type=str, default=32, help='STID_emb_dim')
args = ap.parse_args() 
print('--------Parameters--------')
print(args)
print('--------------------------')

# os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dcrnn_model import *


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

args.cuda = torch.cuda.is_available() 
print(args.cuda)
logger.info('cuda %s', args.cuda)

time_token = str(time.time()).split('.')[0] # tensorboard model
log_token = '%s.%s.w-%s.h-%s.%s' % (args.model, args.dataset, args.window, args.horizon, args.rnn_model)

'''
if args.mylog:
    tensorboard_log_dir = 'tensorboard/%s' % (log_token)
    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)
    writer = SummaryWriter(tensorboard_log_dir)
    shutil.rmtree(tensorboard_log_dir)
    logger.info('tensorboard logging to %s', tensorboard_log_dir)
'''

data_loader = DataBasicLoader(args)


if args.model == 'ISID':
    model = SIDS(args, return_matrix, data_loader)  
else: 
    raise LookupError('can not find the model')
 
logger.info('model %s', model)
if args.cuda:
    model.cuda()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('#params:',pytorch_total_params)

def evaluate(data_loader, data, tag='val'):
    model.eval()
    total = 0.
    n_samples = 0.
    total_loss = 0.
    y_true, y_pred = [], []
    batch_size = args.batch
    y_pred_mx = []
    y_true_mx = []
    for inputs in data_loader.get_batches(data, batch_size, False):
        X, Y = inputs[0], inputs[1]
        X = X.unsqueeze(dim=-1) # reshape to 4-D shape for SID    
        if return_matrix == False:
            output  = model(X)
        if return_matrix == True:
            output, spatial_matrix = model(X)
        loss_train = F.l1_loss(output, Y) # mse_loss
        total_loss += loss_train.item()
        n_samples += (output.size(0) * data_loader.m)

        y_true_mx.append(Y.data.cpu())
        y_pred_mx.append(output.data.cpu())

    y_pred_mx = torch.cat(y_pred_mx)
    y_true_mx = torch.cat(y_true_mx) # [n_samples, 47] 
    y_true_states = y_true_mx.numpy() * (data_loader.max - data_loader.min ) * 1.0 + data_loader.min  
    y_pred_states = y_pred_mx.numpy() * (data_loader.max - data_loader.min ) * 1.0 + data_loader.min  #(#n_samples, 47)
    rmse_states = np.mean(np.sqrt(mean_squared_error(y_true_states, y_pred_states, multioutput='raw_values'))) # mean of 47
    raw_mae = mean_absolute_error(y_true_states, y_pred_states, multioutput='raw_values')
    std_mae = np.std(raw_mae) # Standard deviation of MAEs for all states/places 
    pcc_tmp = []
    for k in range(data_loader.m):
        pcc_tmp.append(pearsonr(y_true_states[:,k],y_pred_states[:,k])[0])
    pcc_states = np.mean(np.array(pcc_tmp)) 
    r2_states = np.mean(r2_score(y_true_states, y_pred_states, multioutput='raw_values'))
    var_states = np.mean(explained_variance_score(y_true_states, y_pred_states, multioutput='raw_values'))

    # convert y_true & y_pred to real data
    y_true = np.reshape(y_true_states,(-1))
    y_pred = np.reshape(y_pred_states,(-1))
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    pcc = pearsonr(y_true,y_pred)[0]
    r2 = r2_score(y_true, y_pred,multioutput='uniform_average') #variance_weighted 
    var = explained_variance_score(y_true, y_pred, multioutput='uniform_average')
    peak_mae = peak_error(y_true_states.copy(), y_pred_states.copy(), data_loader.peak_thold)
    global y_true_t
    global y_pred_t
    y_true_t = y_true_states
    y_pred_t = y_pred_states
    return float(total_loss / n_samples), mae,std_mae, rmse, rmse_states, pcc, pcc_states, r2, r2_states, var, var_states, peak_mae, y_true_t, y_pred_t


 
bad_counter = 0
best_epoch = 0
best_val = 1e+20;
try:
    print('begin training');
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        # train_loss = train(data_loader, data_loader.train)
        
        model.train()
        total_loss = 0.
        n_samples = 0.
        batch_size = args.batch
        for inputs in data_loader.get_batches(data_loader.train, batch_size, False):
            X, Y = inputs[0], inputs[1]
            X = X.unsqueeze(dim=-1) # reshape to 4-D shape for SID    
            optimizer.zero_grad()
            if return_matrix == False:
                output  = model(X)
            if return_matrix == True:
                output, spatial_matrix = model(X)      
            # if Y.size(0) == 1:
            #     Y = Y.view(-1)
            loss_train = F.l1_loss(output, Y) # mse_loss
            total_loss += loss_train.item()
            loss_train.backward()
            optimizer.step()
            n_samples += (output.size(0) * data_loader.m)
        train_loss = float(total_loss / n_samples)
            
        val_loss, mae,std_mae, rmse, rmse_states, pcc, pcc_states, r2, r2_states, var, var_states, peak_mae, y_true_t, y_pred_t = evaluate(data_loader, data_loader.val)
        print('Epoch {:3d}|time:{:5.2f}s|train_loss {:5.8f}|val_loss {:5.8f}'.format(epoch, (time.time() - epoch_start_time), train_loss, val_loss))
        '''
        if args.mylog:
            writer.add_scalars('data/loss', {'train': train_loss}, epoch )
            writer.add_scalars('data/loss', {'val': val_loss}, epoch)
            writer.add_scalars('data/mae', {'val': mae}, epoch)
            writer.add_scalars('data/rmse', {'val': rmse_states}, epoch)
            writer.add_scalars('data/rmse_states', {'val': rmse_states}, epoch)
            writer.add_scalars('data/pcc', {'val': pcc}, epoch)
            writer.add_scalars('data/pcc_states', {'val': pcc_states}, epoch)
            writer.add_scalars('data/R2', {'val': r2}, epoch)
            writer.add_scalars('data/R2_states', {'val': r2_states}, epoch)
            writer.add_scalars('data/var', {'val': var}, epoch)
            writer.add_scalars('data/var_states', {'val': var_states}, epoch)
            writer.add_scalars('data/peak_mae', {'val': peak_mae}, epoch)
        ''' 
        # Save the model if the validation loss is the best we've seen so far.
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            bad_counter = 0
            model_path = '%s/%s.pt' % (args.save_dir, log_token)
            with open(model_path, 'wb') as f:
                torch.save(model.state_dict(), f)
            print('Best validation epoch:',epoch, time.ctime());
            test_loss, mae,std_mae, rmse, rmse_states, pcc, pcc_states, r2, r2_states, var, var_states, peak_mae, y_true_t, y_pred_t  = evaluate(data_loader, data_loader.test,tag='test')
            print('TEST MAE {:5.4f} std {:5.4f} RMSE {:5.4f} RMSEs {:5.4f} PCC {:5.4f} PCCs {:5.4f} R2 {:5.4f} R2s {:5.4f} Var {:5.4f} Vars {:5.4f} Peak {:5.4f}'.format( mae, std_mae, rmse, rmse_states, pcc, pcc_states,r2, r2_states, var, var_states, peak_mae))
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early, epoch',epoch)

# Load the best saved model.
model_path = '%s/%s.pt' % (args.save_dir, log_token)
with open(model_path, 'rb') as f:
    model.load_state_dict(torch.load(f));
test_loss, mae,std_mae, rmse, rmse_states, pcc, pcc_states, r2, r2_states, var, var_states, peak_mae, y_true_t, y_pred_t  = evaluate(data_loader, data_loader.test,tag='test')
print('Final evaluation')
print('TEST MAE {:5.4f} std {:5.4f} RMSE {:5.4f} RMSEs {:5.4f} PCC {:5.4f} PCCs {:5.4f} R2 {:5.4f} R2s {:5.4f} Var {:5.4f} Vars {:5.4f} Peak {:5.4f}'.format( mae, std_mae, rmse, rmse_states, pcc, pcc_states,r2, r2_states, var, var_states, peak_mae))
# y_true_t, y_pred_t = y_true_t.flatten(order='C'), y_pred_t.flatten(order='C')
# from diebold_mariano_test import *
# d_t_list = cul_d_t(MSE, y_true_t, y_pred_t, y_pred_t)
# # d_t_list = cul_d_t(MAE, y_true_t, y_pred_t, y_pred_t)
# cul_DM(d_t_list)
# cul_P(d_t_list)
import matplotlib
import matplotlib.pyplot as plt
if return_matrix == True: #Draw TSNE analysis
    from openTSNE import TSNE
    a = spatial_matrix[1].cpu().detach().numpy().T
    # X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(a)
    X_embedded = TSNE().fit(a)
    plt.scatter(X_embedded[:,0],X_embedded[:,1])
    plt.show()


if return_matrix == False: #Draw SHAP analysis
    import shap
    
    draw_output = 3 # the number of displayed SHAP outputs, maximum = node num = provinces
    SHAP_sample = 1 # the number for SHAP analysis
    
    background = X[:-SHAP_sample]
    test_images = X[-SHAP_sample:] #单样本 输出格式 [1, 47] 输入格式[1, 20, 47, 1]
    
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(test_images)
    
    shap_numpy = [np.swapaxes(s, 1, 2) for s in shap_values]
    test_numpy = np.swapaxes(test_images.cpu().detach().numpy(), 1, 2)
    
    
    # shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    # test_numpy = np.swapaxes(np.swapaxes(test_images.cpu().detach().numpy(), 1, -1), 1, 2)
    print(test_numpy.shape)
    # plot the feature attributions
    prediction_result = model(test_images).cpu().detach().numpy()
    shap.image_plot(shap_numpy[:draw_output], pixel_values = -test_numpy, 
                    labels=prediction_result[:,:draw_output])
    print("Ground Truth: ", Y[-1:])                                                                                                                                                                    