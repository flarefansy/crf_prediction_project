clear; clc;

%input info
features = [  6.17378616e+00,   3.83100014e+01,   3.95950012e+01, 5.22934246e+00,   5.59000015e-01,   8.22000027e-01,5.09000003e-01,   6.25999987e-01,   2.66259995e+01,1.54999995e+00,   1.09999999e-02,   2.21000001e-01, 8.79999995e-01,   2.11899996e+00,   9.40999985e-01,3.00000000e+01]
label = [  0.51584, -13.699  ,  87.687  ]

% pre-processing


% training result weight and biases

load('saveddata1.mat');
load('saveddata2.mat');
load('saveddata3.mat');
load('saveddata4.mat');

load('saveddata8.mat');
load('saveddata9.mat');

load('saveddata11.mat');
load('saveddata12.mat');
load('1st.mat');
load('1st_bn.mat');

% dataset pre-processing info
features_min = [ -6.17378616e+00,  -4.67508618e-01,  -4.94127542e-01,-3.78625657e-01,  -2.89512535e-01,  -7.14028811e-01,-2.69360265e-02,  -2.86628782e-04,  -2.58342303e-03,-5.22937969e-04,  -1.50943407e-02,   0.00000000e+00,-1.80112050e+00,   0.00000000e+00,  -1.11524161e-03,-3.00000000e+01];
features_scale = [  1.00000000e+00,   1.71594276e-02,   1.72464332e-02,1.66413210e-01,   1.47710481e+00,   1.79856117e+00,1.12233443e+00,   4.77714632e-02,   1.26638386e-02,4.75398157e-02,   1.88679250e+00,   1.22850121e+00,2.80112050e+00,   2.72739675e-03,   1.85873601e-01,1.00000000e+00];
label_mean = [  0.54570984, -14.17647008,  90.04360461];
label_std = [  0.57919609,   5.90142651,  13.44028738];

% label prediction from training program
label_pred = [ 0.25868097, -0.26287815, -0.06614494];

% maxminscaler
features_init_mat = features.*features_scale+features_min; %%%%%%%%%%%% question
% eval_features = sum(abs(features_prep - features_init_mat));

% standardscaler
label_init_mat = (label - label_mean)./label_std;
% eval_label = sum(abs(label_prep - label_init_mat));

% input
input = features_init_mat(1,3:15);

% first layer
out_1st = input*weight_1st+biase_1st;
out_1st_ac = tanh(out_1st);
out_1st_bn = (out_1st_ac-mean_1st)./sqrt(var_1st+0.001);
out_1st_out = out_1st_bn.*scale_1st + shift_1st;

% output layer
out = out_1st_out*weight_out+biase_out;
eval_out = sum(abs(out-label_pred));
