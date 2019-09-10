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

% This file used for finding a good parameter combination
% Our algorithm is a scalable extension of SSC [A3]
% [A3] Elhamifar, Ehsan, and Rene Vidal.
%      "Sparse subspace clustering: Algorithm, theory, and applications."
%      Pattern Analysis and Machine Intelligence, IEEE Transactions on 35.11 (2013): 2765-2781.
% **************

close all;
clear all;
clc;

%% --------------------------------------------------------------------------
addpath ('../usage/');
addpath ('../data/');

fprintf('Beginning!\n');
% loading data
CurData = 'YaleB_48_42';
load (CurData);  
load ([CurData '_landmarkID']);
% --- parameters configuration
par.nClass             =   length(unique(labels));
par.Classifier         =   2 ;% 1 for SRC, 2 for CRC, 
par = L1ParameterConfig(par);
% --- creat a dir for store logs and final result
% --- only the data with the belong to the first par.nClass subject are processed

% DATA   =   double(DATA(:,labels<=par.nClass));
% labels     =   labels(labels<=par.nClass);

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
    par.NameStr = ['ASSC_CR+1nn' '_' CurData '_cluster#' num2str(par.nClass) '_EigenF' num2str(par.nDim) '_lambda#' num2str(length(par.lambda)) '_tolerance#' num2str(length(par.tolerance)) '_landmark#' num2str(par.landmarkNO)];
end;
% fid = fopen([par.NameStr '.txt'], 'a+');

for i = 1:length(par.lambda)
    for j = 1:length(par.tolerance)
        fprintf([' \n--------- Running the experiment when lambda = ' num2str(par.lambda(i)) ' | tolerance = ' num2str(par.tolerance(j)) ' ---------\n ']);     
       %  fprintf(fid,[' \n--------- Running the experiment when lambda = ' num2str(par.lambda(i)) ' | tolerance = ' num2str(par.tolerance(j)) ' ---------\n ']);     
        % clustering the landmark
        tic;
        Tr_plabel = InSample(Tr_dat, par.lambda(i), par.tolerance(j), par);        
        time(i,j)=toc;
        % classification the non-landmark
        tic;
        Tt_plabel = OutSample(Tr_dat, Tt_dat, Tr_plabel, par.lambda(i), par.tolerance(j),par);
        time(i,j)=time(i,j)+toc;        
        % --- get accuracy and normalized mutual information for clustering
        % landmark
        P_label = [Tr_plabel Tt_plabel]';
        clear Classifier_plabel Tr_plabel;
        if par.Classifier ~= 0            
            P_label = bestMap(labels,P_label);
            labels = reshape(labels,[],1);
            accuracy(i,j) = length(find(labels == P_label))/length(labels);
            nmi(i,j) = MutualInfo(labels,P_label);
%                 [t_accuracy t_nmi]= CalMetricOfCluster(labels,P_label);
%                 t_accuracy = t_accuracy./100;
%                 accuracy(i,j) = t_accuracy;
%                 nmi(i,j) = t_nmi;
        end;        
        fprintf([' *** the accuracy scores is: ' num2str(accuracy(i)) '\n']);
        fprintf([' + the normalized mutual information score is: ' num2str(nmi(i)) '\n']);  
        fprintf([' - The time cost is : ' num2str(time(i,j)) '\n\n']);  
    end;
end;
% fclose(fid);
clear Tr_dat Tr_gnd Tr_plabel Tt_dat Tt_gnd WTA_plabel SRC_plabel ans fid i j landmark_ID labels P_label Tt_plabel;
save (par.NameStr);
% exit;