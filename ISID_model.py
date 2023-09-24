# -*- coding: utf-8 -*-

import torch
from torch import nn


class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.
        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]
        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        hidden = hidden + input_data                           # residual
        return hidden
    
    
class SIDS(nn.Module):
    """
    The implementation of CIKM 2022 short paper
        "Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting"
    Link: https://arxiv.org/abs/2208.05233
    """

    def __init__(self, args, return_matrix, data):
        super().__init__()
        # attributes
        self.num_nodes = data.dat.shape[1]
        self.node_dim = args.SID_D_dim #embedding dim D
        self.input_len = data.P
        self.input_dim = 1 #only infectious number
        self.embed_dim = args.SID_emb_dim
        self.output_len = 1 # output time series lenth 1
        self.num_layer = args.SID_layer_num
        
        # no use
        self.temp_dim_tid = 32
        self.temp_dim_diw = 32

        self.if_time_in_day = False
        self.if_day_in_week = False
        self.if_spatial = True

        self.return_matrix = return_matrix
        # spatial embeddings
        # S-matrix size： node_num*node_dim(D) for each seq, e.g. batch (128, 20, 47, 1), each S-Matrix (47, D), entire S-matrix (128, 47, D) 
        if self.if_spatial: 
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(7, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(7, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

        # encoding
        self.hidden_dim = self.embed_dim+self.node_dim * \
            int(self.if_spatial)+self.temp_dim_tid*int(self.if_day_in_week) + \
            self.temp_dim_diw*int(self.if_time_in_day)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])

        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)

    def forward(self, history_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """Feed forward of SID.
        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]
        Returns:
            torch.Tensor: prediction wit shape [B, L, N, C]
        """

        # prepare data
        input_data = history_data[..., range(self.input_dim)]
        # print('SID input shape: ' , input_data.shape)
        
        if self.if_time_in_day:
            t_i_d_data = history_data[..., 1]
            time_in_day_emb = self.time_in_day_emb[(
                t_i_d_data[:, -1, :] * 7).type(torch.LongTensor)]
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            d_i_w_data = history_data[..., 2]
            day_in_week_emb = self.day_in_week_emb[(
                d_i_w_data[:, -1, :]).type(torch.LongTensor)]
        else:
            day_in_week_emb = None

        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous() # B, N, L, C
        input_data = input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1) # B, L*C, N, 1
        time_series_emb = self.time_series_emb_layer(input_data) # B, embed_dim, N, 1

        node_emb = []
        if self.if_spatial:
            # expand node embeddings
            # self.node_emb size: node_num*feature_size
            node_emb.append(self.node_emb.unsqueeze(0).expand(
                batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)) # B, D, N, 1
        # temporal embeddings
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        # concate all embeddings
        # print("time_series_emb:" , time_series_emb.shape)
        # print("node_emb:" , node_emb[0].shape)
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)
        # print("After concat:" , hidden.shape)

        # encoding
        hidden = self.encoder(hidden)

        # regression
        prediction = self.regression_layer(hidden)
        
        # for EpiGNN
        prediction = prediction.squeeze(1)
        prediction = prediction.squeeze(-1)
        # print('SID prediction shape: ', prediction.shape)
        
        node_emb = torch.stack(node_emb)
        node_emb = node_emb.squeeze(0)
        node_emb = node_emb.squeeze(-1)
        # print('Spatial node embedding shape: ', node_emb.shape)
        
        if self.return_matrix == True:
            return prediction, node_emb
        if self.return_matrix == False:
            return prediction
        # return prediction