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

% This file is the code of LSR [A3],i.e.,
% [A3] Lu, Can-Yi, et al.
%     "Robust and efficient subspace segmentation via least squares regression."
%      Computer Vision–ECCV 2012. Springer Berlin Heidelberg, 2012. 347-360.
%% **************

function Z = LSR1( X , lambda )

%--------------------------------------------------------------------------
% Copyright @ Can-Yi Lu, 2012
%--------------------------------------------------------------------------

% Input
% X             Data matrix, dim * num
% lambda        parameter, lambda>0


% Output the solution to the following problem:
% min ||X-XZ||_F^2+lambda||Z||_F^2
%   s.t. diag(Z)=0

% Z             num * num

if nargin < 2
    lambda = 0.001 ;
end
[dim,num] = size(X) ;


% for i = 1 : num
%    X(:,i) = X(:,i) / norm(X(:,i)) ; 
% end


I = eye(num) ;
invX = I / (X'*X+lambda*I) ;
Z = zeros( num , num ) ;
for i = 1 : num
    Z(:,i) = invX(:,i) / invX(i,i) ;
    Z(i,i) = 0 ;
end
Z = -1 * Z ;