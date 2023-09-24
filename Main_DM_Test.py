# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


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
from dcrnn_model import *
from models import *

import shutil
import logging
import glob
import time
from tensorboardX import SummaryWriter

import pandas as pd
from diebold_mariano_test import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp

# Training settings
ap = argparse.ArgumentParser()
# ap.add_argument('--dataset', type=str, default='japan', help="Dataset string")
# ap.add_argument('--sim_mat', type=str, default='japan-adj', help="adjacency matrix filename (*-adj.txt)")
ap.add_argument('--dataset', type=str, default='region785', help="Dataset string")
ap.add_argument('--sim_mat', type=str, default='region-adj', help="adjacency matrix filename (*-adj.txt)")

ap.add_argument('--n_layer', type=int, default=1, help="number of layers (default 1)") 
ap.add_argument('--n_hidden', type=int, default=20, help="rnn hidden states (could be set as any value)") 
ap.add_argument('--seed', type=int, default=42, help='random seed')
ap.add_argument('--epochs', type=int, default=50, help='1500 default number of epochs to train')
ap.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
ap.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
ap.add_argument('--dropout', type=float, default=0.2, help='dropout rate usually 0.2-0.5.')
ap.add_argument('--batch', type=int, default=32, help="batch size")
ap.add_argument('--check_point', type=int, default=1, help="check point")
ap.add_argument('--shuffle', action='store_true', default=False, help="not used, default false")
ap.add_argument('--train', type=float, default=.6, help="Training ratio (0, 1)")
ap.add_argument('--val', type=float, default=.2, help="Validation ratio (0, 1)")
ap.add_argument('--test', type=float, default=.2, help="Testing ratio (0, 1)")

ap.add_argument('--model', default='SID', choices=['cola_gnn','CNNRNN_Res','RNN',
                                                   'AR','ARMA','VAR','GAR','SelfAttnRNN',
                                                   'lstnet','stgcn','dcrnn',
                                                   'SIDS', 'SID'], help='')

ap.add_argument('--rnn_model', default='RNN', choices=['LSTM','RNN','GRU'], help='')
ap.add_argument('--mylog', action='store_false', default=True,  help='save tensorboad log')
ap.add_argument('--cuda', action='store_true', default=True,  help='')
ap.add_argument('--window', type=int, default=20, help='')
ap.add_argument('--horizon', type=int, default=10, help='leadtime default 1')
ap.add_argument('--save_dir', type=str,  default='./src/save',help='dir path to save the final model')
ap.add_argument('--gpu', type=int, default=1,  help='choose gpu 0-10')
ap.add_argument('--lamda', type=float, default=0.01,  help='regularize params similarities of states')
ap.add_argument('--bi', action='store_true', default=False,  help='bidirectional default false')
ap.add_argument('--patience', type=int, default=200, help='patience default 100')
ap.add_argument('--k', type=int, default=10,  help='kernels')
ap.add_argument('--hidsp', type=int, default=15,  help='spatial dim')
ap.add_argument('--SID_D_dim', type=str, default=32, help='STID_D_dim')
ap.add_argument('--SID_layer_num', type=str, default=2, help='STID_layer_num')
ap.add_argument('--SID_emb_dim', type=str, default=32, help='STID_emb_dim')
args = ap.parse_args() 
print('--------Parameters--------')
print(args)
print('--------------------------')

# os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

args.cuda = torch.cuda.is_available() 
print(args.cuda)
logger.info('cuda %s', args.cuda)

for lead_time in [3]:
    args.horizon = lead_time
    
    time_token = str(time.time()).split('.')[0] # tensorboard model
    log_token = '%s.%s.w-%s.h-%s.%s' % (args.model, args.dataset, args.window, args.horizon, args.rnn_model)
    
    
    data_loader = DataBasicLoader(args)
    
    return_matrix = False
    model_selections = ['AR','ARMA','VAR','GAR', 'RNN','ATTRNN', 'DCRNN',
        'LSTNet','STGCN','Cola-GNN',
        'ISID', 'ISID-wo']
    model_total_results = pd.DataFrame()
    
    for model_name in model_selections:
        args.model = model_name
    
        if args.model == 'ISID':
            model = SIDS(args, return_matrix, data_loader)  
        elif args.model == 'ISID-wo':
            model = SID(args, data_loader)  
        elif args.model == 'CNNRNN_Res':
            model = CNNRNN_Res(args, data_loader)  
        elif args.model == 'RNN':
            model = RNN(args, data_loader)
        elif args.model == 'AR':
            model = AR(args, data_loader)
        elif args.model == 'ARMA':
            model = ARMA(args, data_loader)
        elif args.model == 'VAR':
            model = VAR(args, data_loader)
        elif args.model == 'GAR':
            model = GAR(args, data_loader)
        elif args.model == 'ATTRNN':
            model = SelfAttnRNN(args, data_loader)
        elif args.model == 'LSTNet':
            model = LSTNet(args, data_loader)      
        elif args.model == 'STGCN':
            model = STGCN(args, data_loader, data_loader.m, 1, args.window, 1)  
        elif args.model == 'DCRNN':
            model = DCRNNModel(args, data_loader)   
        elif args.model == 'Cola-GNN':
            model = cola_gnn(args, data_loader)        
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
                
                if args.model == 'ISID' or args.model =='ISID-wo':
                    X = X.unsqueeze(dim=-1) # reshape to 4-D shape for SID    
                    if return_matrix == False:
                        output  = model(X)
                    if return_matrix == True:
                        output, spatial_matrix = model(X)
                elif args.model == 'Cola-GNN':
                    output  = model(X)
                else: 
                    output  = model(X)[0]
                    
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
                    optimizer.zero_grad()
                    if args.model == 'ISID' or args.model =='ISID-wo':
                        X = X.unsqueeze(dim=-1) # reshape to 4-D shape for SID    
                        if return_matrix == False:
                            output  = model(X)
                        if return_matrix == True:
                            output, spatial_matrix = model(X)
                    elif args.model == 'Cola-GNN':
                        output  = model(X)
                    else: 
                        output  = model(X)[0]
                    if Y.size(0) == 1:
                        Y = Y.view(-1)
                    loss_train = F.l1_loss(output, Y) # mse_loss
                    total_loss += loss_train.item()
                    loss_train.backward()
                    optimizer.step()
                    n_samples += (output.size(0) * data_loader.m)
                train_loss = float(total_loss / n_samples)
                    
                val_loss, mae,std_mae, rmse, rmse_states, pcc, pcc_states, r2, r2_states, var, var_states, peak_mae, y_true_t, y_pred_t = evaluate(data_loader, data_loader.val)
                # print('Epoch {:3d}|time:{:5.2f}s|train_loss {:5.8f}|val_loss {:5.8f}'.format(epoch, (time.time() - epoch_start_time), train_loss, val_loss))
        
                # Save the model if the validation loss is the best we've seen so far.
                if val_loss < best_val:
                    best_val = val_loss
                    best_epoch = epoch
                    bad_counter = 0
                    model_path = '%s/%s.pt' % (args.save_dir, log_token)
                    with open(model_path, 'wb') as f:
                        torch.save(model.state_dict(), f)
                    # print('Best validation epoch:',epoch, time.ctime());
                    test_loss, mae,std_mae, rmse, rmse_states, pcc, pcc_states, r2, r2_states, var, var_states, peak_mae, y_true_t, y_pred_t  = evaluate(data_loader, data_loader.test,tag='test')
                    # print('TEST MAE {:5.4f} std {:5.4f} RMSE {:5.4f} RMSEs {:5.4f} PCC {:5.4f} PCCs {:5.4f} R2 {:5.4f} R2s {:5.4f} Var {:5.4f} Vars {:5.4f} Peak {:5.4f}'.format( mae, std_mae, rmse, rmse_states, pcc, pcc_states,r2, r2_states, var, var_states, peak_mae))
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
        y_true_t, y_pred_t = y_true_t.flatten(order='C'), y_pred_t.flatten(order='C')
        model_total_results[model_name+'True'] = y_true_t
        model_total_results[model_name+'Pred'] = y_pred_t
   ### Final DM test
    DM_compare = model_selections
    DM_pvalue = model_selections 
    y_true_t = model_total_results[model_selections[0]+'True']
    for compare_i in range(len(model_selections)): #n*n models comparations
        DM_compare_i = []
        DM_pvalue_i = []
        for compare_j in range(len(model_selections)):
            y_pred_t1 = model_total_results[model_selections[compare_i]+'Pred']
            y_pred_t2 = model_total_results[model_selections[compare_j]+'Pred']
            d_t_list = cul_d_t(MSE, y_true_t, y_pred_t1, y_pred_t2) # 结果矩阵aij为i和j比
            DM_compare_i.append(cul_DM(d_t_list))
            DM_pvalue_i.append(cul_P(d_t_list))
        DM_compare = np.row_stack(( DM_compare, DM_compare_i ))
        DM_pvalue = np.row_stack(( DM_pvalue, DM_pvalue_i ))
        
    from dm_test import dm_test
    DM_compare = model_selections
    DM_pvalue = model_selections 
    y_true_t = model_total_results[model_selections[0]+'True']
    for compare_i in range(len(model_selections)): #n*n models comparations
        DM_compare_i = []
        DM_pvalue_i = []
        for compare_j in range(len(model_selections)):
            y_pred_t1 = model_total_results[model_selections[compare_i]+'Pred']
            y_pred_t2 = model_total_results[model_selections[compare_j]+'Pred']
            d_t_list = dm_test(y_true_t,y_pred_t1,y_pred_t2,h = 3, crit="MAD") # 结果矩阵aij为i和j比
            DM_compare_i.append(d_t_list.DM)
            # DM_pvalue_i.append(cul_P(d_t_list))
        DM_compare = np.row_stack(( DM_compare, DM_compare_i ))
        # DM_pvalue = np.row_stack(( DM_pvalue, DM_pvalue_i ))
        
    # dm_test(y_true_t,y_pred_t1,y_pred_t2,h = 3, crit="MAD")
    
    # #  DM values  
    # DM_compare=pd.DataFrame(DM_compare[1:, :], columns=DM_compare[0, :])
    # np.fill_diagonal(DM_compare.values, 0)
    
    #  DM value heatmap
    import matplotlib.pyplot as plt
    import seaborn as sns
    DM_compare=pd.DataFrame(DM_compare[1:, :], columns=DM_compare[0, :])
    np.fill_diagonal(DM_compare.values, 0)
    DM_compare = DM_compare.astype(float)
    DM_compare.index = DM_compare.columns.values
    fig1, ax1 = plt.subplots(figsize=(20,20)) 
    ax1 = sns.heatmap(DM_compare, annot= True,fmt=".2f",
                      linewidths=0.005,linecolor="grey",
                      cbar=(False)
                      )
    DM_name = './results/' + time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()) + '_' + args.dataset + '_window_' + str(args.window) + '_leadtime_' + str(args.horizon) + '_epoch_' + str(args.epochs) + '_batchsize_' + str(args.batch) + '_trainset_' + str(args.train) + '_DM_MAE.png'
    fig1.savefig(DM_name, dpi=330, bbox_inches = 'tight') 
            # print('MSE DM: ', cul_DM(d_t_list))
            # print('P-value DM: ', cul_P(d_t_list))
            # p1 p2  H0: p1--p2 h1: p1 // p2 p<0.05 
            # p值小于显著性水平应该拒绝原假设，反之则不拒绝
