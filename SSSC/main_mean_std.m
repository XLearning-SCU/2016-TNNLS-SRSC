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

% This file performs multiple runs to obtain mean and std accuracy.
% Our algorithm is a scalable extension of SSC [A3]
% [A3] Elhamifar, Ehsan, and Rene Vidal.
%      "Sparse subspace clustering: Algorithm, theory, and applications."
%      Pattern Analysis and Machine Intelligence, IEEE Transactions on 35.11 (2013): 2765-2781.
%% **************

close all;
clear all;
clc;

%% --------------------------------------------------------------------------
addpath ('../usage/');
addpath ('../data/');

fprintf('Beginning!\n');
% loading data
CurData = 'Reuters21578_PCA85';
load (CurData);  
load ([CurData '_landmarkID']);
% --- parameters configuration
par.nClass             =   length(unique(labels));
par.Classifier         =   2;% 1 for SRC, 2 for CRC, 
par.nDim               =   785; % the eigenfaces dimension
par.landmarkNO         =   2000; % the number of landmark or representative data
% --- optimization algorithms configuration
par.lambda             =    [1e-7];
par.tolerance          =    [0.01];

par.maxIteration = 2000;
par.isNonnegative = false;
STOPPING_GROUND_TRUTH = -1;
STOPPING_DUALITY_GAP = 1;
STOPPING_SPARSE_SUPPORT = 2;
STOPPING_OBJECTIVE_VALUE = 3;
STOPPING_SUBGRADIENT = 4;
par.stoppingCriterion = STOPPING_OBJECTIVE_VALUE;
par.SCidx              =   3; % 0-test three spectral clusterings, 1-only Unnormalized Method, 2-Random Walk Method and 3-Normalized Symmetric



% --- dimension reduction using PCA
DATA = double(DATA);
labels = double(labels);
dat = FeatureEx(DATA, par);
clear DATA;

% --- split the data into two parts for landmark and non-landmark
Tr_dat = dat(:,landmark_ID(1:par.landmarkNO)); % the landmark data;
Tt_dat = dat(:,landmark_ID(1+par.landmarkNO:end)); % the non-landmark data;
labels = labels(landmark_ID);
clear dat;

if par.Classifier == 1
    par.NameStr = ['SSSC_SRC' '_' CurData '_cluster#' num2str(par.nClass) '_EigenF' num2str(par.nDim) '_lambda#' num2str(length(par.lambda)) '_tolerance#' num2str(length(par.tolerance)) '_landmark#' num2str(par.landmarkNO)];
elseif par.Classifier == 2
    par.NameStr = ['SSSC_CRC' '_' CurData '_cluster#' num2str(par.nClass) '_EigenF' num2str(par.nDim) '_lambda#' num2str(length(par.lambda)) '_tolerance#' num2str(length(par.tolerance)) '_landmark#' num2str(par.landmarkNO)];
elseif par.Classifier == 3
    par.NameStr = ['SSSC_CR+1nn' '_' CurData '_cluster#' num2str(par.nClass) '_EigenF' num2str(par.nDim) '_lambda#' num2str(length(par.lambda)) '_tolerance#' num2str(length(par.tolerance)) '_landmark#' num2str(par.landmarkNO)];
end;
% fid = fopen([par.NameStr '.txt'], 'a+');

for i = 1:5
        fprintf([' \n--------- Running the experiment when lambda = ' num2str(par.lambda) ' | tolerance = ' num2str(par.tolerance) ' ---------\n ']);     
       %  fprintf(fid,[' \n--------- Running the experiment when lambda = ' num2str(par.lambda(i)) ' | tolerance = ' num2str(par.tolerance(j)) ' ---------\n ']);     
        % clustering the landmark
        tic;
        Tr_plabel = InSample(Tr_dat, par.lambda, par.tolerance, par);        
        time(i)=toc;
        % classification the non-landmark
        tic;
        Tt_plabel = OutSample(Tr_dat, Tt_dat, Tr_plabel, par.lambda, par.tolerance,par);
        time(i)=time(i)+toc;        
        % --- get accuracy and normalized mutual information for clustering
        % landmark
        Tr_plabel = reshape(Tr_plabel,1,[]);
        Tt_plabel = reshape(Tt_plabel,1,[]);
        P_label = [Tr_plabel Tt_plabel]';
        P_label = reshape(P_label,[],1);
        
        P_label = bestMap(labels,P_label);
        labels = reshape(labels,[],1);
        accuracy(i) = length(find(labels == P_label))/length(labels);
        nmi(i) = MutualInfo(labels,P_label);
end;
% fclose(fid);
clear Tr_dat Tr_gnd Tr_plabel Tt_dat Tt_gnd WTA_plabel SRC_plabel ans fid i j landmark_ID labels P_label Tt_plabel;
clear Classifier_plabel Tr_plabel;

save (par.NameStr);
clc;
fprintf([' | the result of SSSC for lambda = ' num2str(par.lambda) '   ' CurData '\n']);
fprintf([' + The mean accuracy is: ' num2str(mean(accuracy*100)) ', and the std is : ' num2str(std(accuracy*100)) '\n']);
fprintf([' + The mean nmi      is: ' num2str(mean(nmi*100)) ', and the std is : ' num2str(std(nmi*100)) '\n']);
fprintf([' + The mean time     is: ' num2str(mean(time)) ', and the std is : ' num2str(std(time)) '\n']);
