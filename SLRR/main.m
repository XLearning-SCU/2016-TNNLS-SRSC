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

% This file used for finding a good parameter combination.
% Our algorithm is a scalable extension of LRR [A3]
% [A3] Liu, Guangcan, Zhouchen Lin, and Yong Yu.
%      "Robust subspace segmentation by low-rank representation."
%       Proceedings of the 27th international conference on machine learning (ICML-10). 2010.
%% **************


close all;
clear all;
clc;

%% --------------------------------------------------------------------------
addpath ('../usage/');
addpath ('../data/');

fprintf('Beginning!\n');
% loading data
CurData = 'AR_55_40';
load (CurData);  
load ([CurData '_landmarkID']);

% parameters configuration
par.nClass             =   length(unique(labels));
par.nDim               =   167; % the eigenfaces dimension
par.landmarkNO         =   700;
% par.lambda             =   [0.1:0.2:4];
par.lambda             =   [3.1];
par.kappa              =   [1e-6]; % the parameter for crc
par.SCidx              =   3; % 0-test three spectral clusterings, 1-only Unnormalized Method, 2-Random Walk Method and 3-Normalized Symmetric
% creat a dir for store logs and final result
par.NameStr = ['SLRR_CRC' '_' CurData '_cluster#' num2str(par.nClass) '_EigenF' num2str(par.nDim) '_lambda#' num2str(length(par.lambda)) '_kappa#' num2str(length(par.kappa)) '_landmarkNO#' num2str(par.landmarkNO)];
% --- split DAT into two parts, using the first par.nClass class to test
DATA = double(DATA);
labels = double(labels);

dat = FeatureEx(DATA, par);
labels=labels(landmark_ID);
clear DATA ;

K = max(labels);
for i = 1:length(par.lambda)
    % clustering on in-samples    
    tic;
    Tr_label=InSample(dat(:,landmark_ID(1:par.landmarkNO)), par.lambda(i), par);
    t_time1 = toc;
    fprintf([' *** The time cost for clustering insample is ' num2str(t_time1) ' seconds, where | lambda = ' num2str(par.lambda(i)) ' \n']);
    % classification the non-landmark
    Tr_label=Tr_label';
    for j=1:length(par.kappa)
        tic;
        Tt_label = OutSample(dat(:,landmark_ID(1:par.landmarkNO)), dat(:,landmark_ID(1+par.landmarkNO:end)), Tr_label, par.kappa(j));
        time(i,j)=t_time1+toc;
        Tr_label = reshape(Tr_label,1,[]);
        Tt_label = reshape(Tt_label,1,[]);
        P_label = [Tr_label Tt_label]';
        P_label = reshape(P_label,[],1);
        P_label = bestMap(labels,P_label);
        labels = reshape(labels,[],1);
        accuracy(i,j) = length(find(labels == P_label))/length(labels);
        nmi(i,j) = MutualInfo(labels,P_label);
        
        fprintf([' | the result for lambda = ' num2str(par.lambda(i)) '\n']);
        fprintf([' + The accuracy                      scores are: ' num2str(accuracy(i,j)) '\n']);
        fprintf([' + The normalized mutual information scores are: ' num2str(nmi(i)) '\n']);

    end;
end;
clear tElapsed fid ans Predict_label coef1 coef2 kk trls ttls t_time1;
clear i j pos t_accuracy t_nmi dat labels Tr_label Tt_label K P_label landmark_ID;
save (par.NameStr);
