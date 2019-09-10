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
fprintf ([ '=============for ' par.NameStr(3:end) '=============\n']);

[val idx]= sort(accuracy, 'descend');
fprintf('                              rank1 \t rank2 \t rank3 \n');
fprintf(['The best 3 accuracy are   : ' num2str(val(1)) '\t | ' num2str(val(2)) '\t | ' num2str(val(3)) ...
         '\nthe corresponding nmi is  : ' num2str(nmi(idx(1))) '\t | ' num2str(nmi(idx(2))) '\t | ' num2str(nmi(idx(3)))  ...
          '\nthe corresponding time is  : ' num2str(time(idx(1))) '\t | ' num2str(time(idx(2))) '\t | ' num2str(time(idx(3)))  ...
         '\nwhen lambda =             : ' num2str(par.lambda(idx(1))) '\t | ' num2str(par.lambda(idx(2)))  '\t | ' num2str(par.lambda(idx(3))) '\n\n']);

[val idx]= sort(nmi, 'descend');
fprintf('                                   rank1 \t rank2 \t rank3 \n');
fprintf(['The best 3 nmi are             : ' num2str(val(1)) '\t | ' num2str(val(2)) '\t | ' num2str(val(3)) ...
         '\nthe corresponding accuracy is  : ' num2str(accuracy(idx(1))) '\t | ' num2str(accuracy(idx(2))) '\t | ' num2str(accuracy(idx(3)))  ...
         '\nwhen lambda =                  : ' num2str(par.lambda(idx(1))) '\t | ' num2str(par.lambda(idx(2)))  '\t | ' num2str(par.lambda(idx(3))) '\n\n']);

