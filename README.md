## Code for "Interpretable Spatial IDentity Neural Network-based Epidemic Prediction"
### Abstact
Focusing on epidemic infectious number forecasting, in this paper, we streamlines the well-known Spatio-Temporal IDentity model (STID) and retains the optional spatial identity matrix for learning the contagion relationship between regions. The famous SHapley Additive exPlanations (SHAP) method was adopted to interpret how the model predicts with multivariate sliding-window time series input data. The light model is named as Interpretable Spatial IDentity (ISID) Neural Network. The experimental results are compared with models such as Cola-GNN.
### Installation
`pip install -r requirments.txt`
### Files for reproducing experimental results
- **Main_cross_validation.py**: Main program containing time series cross validation.
- **Main_DM_Test.py**: DM-test for models when prediction ahead step equals to 3.
- **ISID_SHAP.py**: SHAP analysis for ISID. By setting `return_matrix = True`, it is also able to use the T-SNE visualization.
- **Cola_GNN_SHAP.py**: SHAP analysis for Cola-GNN. 

### Related works
**This study was largely inspired by the following two papers, thanks for their outstanding programs**
- Shao, Z., Zhang, Z., Wang, F., Wei, W., & Xu, Y. (2022). Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting. Proceedings of the 31st ACM International Conference on Information & Knowledge Management, 4454–4458. https://github.com/zezhishao/STID
- Deng, S., Wang, S., Rangwala, H., Wang, L., & Ning, Y. (2020). Cola-GNN: Cross-location Attention based Graph Neural Networks for Long-term ILI Prediction. Proceedings of the 29th ACM International Conference on Information & Knowledge Management, 245–254. https://github.com/amy-deng/colagnn

