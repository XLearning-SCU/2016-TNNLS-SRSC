%% **************
% the code is provided to repduce our result reported in our paper [A2]. NOTE THAT [A2] is a journal extension of our AAAI'15 paper [A1].

% [A1] Xi Peng, Zhang Yi, and Huajin Tang,
%      Robust Subspace Clustering via Thresholding Ridge Regression,
%      The Twenty-Ninth AAAI Conference on Artificial Intelligence (AAAI), Austin, Texas, USA, January 25–29, 2015.
% [A2]Xi Peng, Huajin Tang, Lei Zhang, Zhang Yi, and Shijie Xiao,
%     A Unified Framework for Representation-based Subspace Clustering of Out-of-sample and Large-scale Data,
%     IEEE Trans. Neural Networks and Learning Systems, accepted.

% If the codes or data sets are helpful to you, please appropriately CITE our works. Thank you very much!

% More materials can be found from my website:
%            www.pengxi.me,
%     email: pangsaai [at] gmail [dot] com

% This file is used to config the parameters
%% **************



function par = L1ParameterConfig(par)
% --------- data configuration
par.nDim               =   114; % the eigenfaces dimension
par.landmarkNO         =   1212; % the number of landmark or representative data
% --- optimization algorithms configuration
% par.lambda             =    [1e-7 1e-6 1e-5]; % balance factor
par.lambda             =   [1e-3];
par.tolerance          =    [1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 0.1:0.1:0.9];
% par.tolerance         =   [0.5];
par.maxIteration = 100;
par.isNonnegative = false;
STOPPING_GROUND_TRUTH = -1;
STOPPING_DUALITY_GAP = 1;
STOPPING_SPARSE_SUPPORT = 2;
STOPPING_OBJECTIVE_VALUE = 3;
STOPPING_SUBGRADIENT = 4;
par.stoppingCriterion = STOPPING_OBJECTIVE_VALUE;

% clustering algorithm configuration
par.SCidx              =   3; % 0-test three spectral clusterings, 1-only Unnormalized Method, 2-Random Walk Method and 3-Normalized Symmetric

