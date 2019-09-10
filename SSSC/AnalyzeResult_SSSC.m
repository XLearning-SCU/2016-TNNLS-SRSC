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

% This file used for anaylzing performance
%% **************


clc;
best_ac=0;
idx = 0;
lam_idx = 0;
tol_idx = 0;
for i =1:length(par.lambda)
    if best_ac<max(accuracy(i,:))
        [best_ac idx]=max(accuracy(i,:));
        corNMI = nmi(idx);
        corT = time(idx);
        lam_idx = i;
        tol_idx = idx;
    end;
end
classifier = {'SRC' 'CRC'};
fprintf(['=======' CurData '| #landmark=' num2str(par.landmarkNO) ' | using ' classifier{par.Classifier} '\n']);
fprintf(['accuracy ===== NMI ===== time cost\n']);
fprintf([num2str([best_ac corNMI corT]) '\n'])
fprintf(['when lambda = ' num2str(par.lambda(lam_idx)) ', tol = ' num2str(num2str(par.tolerance(tol_idx))) '\n\n']);