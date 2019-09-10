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

% This file is used to clustering in-sample data
%% **************


function Predict_label=InSample(Tr_dat, lambda, par)

Z = LSR1( Tr_dat , lambda );
L = (abs(Z')+abs(Z))./2;
% run spectral clustering
[Predict_label, SingVals, LapKernel] = SpectralClustering(L,par.nClass,par.SCidx);

% if the value is nan, then randomly assign a label, maybe a
    % more suitable strategy is to reject label assignment as it is a outlier
for ii = 1:size(Predict_label,1)
    for jj = 1:size(Predict_label,2)
        if isnan(Predict_label(ii,jj))==1
            tmp = unique([Predict_label]);
            tmp2= tmp(randperm(length(tmp)));
            Predict_label(ii,jj) = tmp2(1);
        end;
    end
end;
Predict_label = reshape(Predict_label,1,[]);
clear ii jj tmp tmp2;