clear; clc;

%input info
features = [6.17378616e+00, 3.75159988e+01, 3.92570000e+01, 5.31669760e+00,6.83000028e-01, 6.97000027e-01, 7.45999992e-01, 8.65999997e-01,6.27899981e+00, 3.60999990e+00, 2.40000002e-02, 1.97999999e-01,9.49999988e-01, 9.30000007e-01, 6.16999984e-01, 3.00000000e+01]

% pre-processing
feature_i = [ 0.        ,  0.17624445,  0.18291569,  0.50614306,  0.71935009,0.53956837,  0.81032545,  0.04108346,  0.07693282,  0.17109579,0.03018868,  0.24324324,  0.85994394,  0.00253648,  0.11356877,  0.        ];

% training result weight and biases

load('saveddata1.mat');
load('saveddata2.mat');
load('saveddata3.mat');
load('saveddata4.mat');
load('saveddata5.mat');
load('saveddata6.mat');


% dataset pre-processing info
features_min = [ -6.17378616e+00,  -4.67508618e-01,  -4.94127542e-01,-3.78625657e-01,  -2.89512535e-01,  -7.14028811e-01,-2.69360265e-02,  -2.86628782e-04,  -2.58342303e-03,-5.22937969e-04,  -1.50943407e-02,   0.00000000e+00,-1.80112050e+00,   0.00000000e+00,  -1.11524161e-03,-3.00000000e+01];
features_scale = [  1.00000000e+00,   1.71594276e-02,   1.72464332e-02,1.66413210e-01,   1.47710481e+00,   1.79856117e+00,1.12233443e+00,   4.77714632e-02,   1.26638386e-02,4.75398157e-02,   1.88679250e+00,   1.22850121e+00,2.80112050e+00,   2.72739675e-03,   1.85873601e-01,1.00000000e+00];
label_mean = [  0.54570984, -14.17647008,  90.04360461];
label_std = [  0.57919609,   5.90142651,  13.44028738];

% label prediction from training program
label_pred = [  0.65886873, -15.580146  ,  90.30154   ];
% maxminscaler
features_init_mat = features.*features_scale+features_min; 

% input
input = features_init_mat(1,3:15);

out_1st = input*weight_1st+biase_1st;
out_1st_in = (out_1st-mean_1st)./sqrt(var_1st+0.001);
out = out_1st_in.*scale_1st + shift_1st;
out_put = out.*label_std + label_mean;
eval_out = sum(abs(out_put-label_pred));

