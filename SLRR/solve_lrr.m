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

% This file is the code of LRR [A3],i.e.,

% [A3] Liu, Guangcan, Zhouchen Lin, and Yong Yu.
%      "Robust subspace segmentation by low-rank representation."
%       Proceedings of the 27th international conference on machine learning (ICML-10). 2010.
%% **************

function [Z,E] = solve_lrr(X,lambda)
Q = orth(X');
A = X*Q;
[Z,E] = lrra(X,A,lambda);
Z = Q*Z;