clear; clc;

%input info
features = [6.17378616e+00,   3.75159988e+01,   3.92570000e+01,5.31669760e+00,   6.83000028e-01,   6.97000027e-01,7.45999992e-01,   8.65999997e-01,   6.27899981e+00,3.60999990e+00,   2.40000002e-02,   1.97999999e-01,9.49999988e-01,   9.30000007e-01,   6.16999984e-01,3.00000000e+01];
label = [  3.37860000e-01,  -1.20480000e+01,   8.45070000e+01   ];

% pre-processing
features_prep = [ 0.        ,  0.17624445,  0.18291569,  0.50614306,  0.71935009,0.53956837,  0.81032545,  0.04108346,  0.07693282,  0.17109579,0.03018868,  0.24324324,  0.85994394,  0.00253648,  0.11356877,  0.        ]
label_prep = [-0.35885919,  0.36067044, -0.41194094];

% training result weight and biases
load('saveddata6.mat');
load('saveddata1.mat');
load('saveddata2.mat');
load('saveddata3.mat');
load('saveddata4.mat');
load('saveddata5.mat');

% dataset pre-processing info
features_min = [ -6.17378616e+00,  -4.67508618e-01,  -4.94127542e-01,-3.78625657e-01,  -2.89512535e-01,  -7.14028811e-01,-2.69360265e-02,  -2.86628782e-04,  -2.58342303e-03,-5.22937969e-04,  -1.50943407e-02,   0.00000000e+00,-1.80112050e+00,   0.00000000e+00,  -1.11524161e-03,-3.00000000e+01];
features_scale = [  1.00000000e+00,   1.71594276e-02,   1.72464332e-02,1.66413210e-01,   1.47710481e+00,   1.79856117e+00,1.12233443e+00,   4.77714632e-02,   1.26638386e-02,4.75398157e-02,   1.88679250e+00,   1.22850121e+00,2.80112050e+00,   2.72739675e-03,   1.85873601e-01,1.00000000e+00];
label_mean = [  0.54570984, -14.17647008,  90.04360461];
label_std = [  0.57919609,   5.90142651,  13.44028738];

% label prediction from training program
label_pred = [1.82633959e-02,   5.97925298e-03,   5.12800179e-05];
label_pred_re = [0.55628794, -14.14118385,  90.04429626];

% maxminscaler
features_init_mat = (features + features_min).*features_scale; %%%%%%%%%%%% question
eval_features = sum(features_prep - features_init_mat);

% standardscaler
label_init_mat = (label - label_mean)./label_std;
eval_label = sum(label_prep - label_init_mat);

% input
input = features_prep(1,3:15);

% first layer
out_1st = input*weight_1st+biase_1st;
out_1st_ac = tanh(out_1st);

% second layer
out_2nd = out_1st_ac*weight_2st+biase_2st;
out_2nd_ac = tanh(out_2nd);

% output layer
out = out_2nd_ac*weight_out+biase_out;

% re-processing
output = out.*label_std + label_mean;
eval_out = sum(output-label_pred_re);
