%--------------------------------------------------------------------------
% This function takes a NxN matrix CMat as adjacency of a graph and 
% computes the segmentation of data from spectral clustering.
% CMat: NxN adjacency matrix
% n: number of groups for segmentation
% K: number of largest coefficients to choose from each column of CMat
% Grps: [grp1,grp2,grp3] for three different forms of Spectral Clustering
% SingVals: [SV1,SV2,SV3] singular values for three different forms of SC
% LapKernel(:,:,i): n last columns of kernel of laplacian to apply KMeans
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2010
%--------------------------------------------------------------------------


function [Grps , SingVals, LapKernel] = SpectralClustering(CKSym,n,SCidx)

N = size(CKSym,1);
MAXiter = 2000; % Maximum iteration for KMeans Algorithm
REPlic = 200; % Replication for KMeans Algorithm

% Method 1: Unnormalized Method
if SCidx==1||SCidx==0
    DKU = diag( sum(CKSym) );
    LapKU = DKU - CKSym;
    [uKU,sKU,vKU] = svd(LapKU);
    f = size(vKU,2);
    kerKU = vKU(:,f-n+1:f);
    svalKU = diag(sKU);
    group1 = kmeans(kerKU,n,'start','sample','maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
end
% Method 2: Random Walk Method
if SCidx==2||SCidx==0
    DKN=( diag( sum(CKSym) ) )^(-1);
    LapKN = speye(N) - DKN * CKSym;
    [uKN,sKN,vKN] = svd(LapKN);
    f = size(vKN,2);
    kerKN = vKN(:,f-n+1:f);
    svalKN = diag(sKN);
    group2 = kmeans(kerKN,n,'start','sample','maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
end;

% Method 3: Normalized Symmetric
if SCidx==3||SCidx==0
    DKS = ( diag( sum(CKSym) ) + 1e-12 )^(-1/2);
    LapKS = speye(N) - DKS * CKSym * DKS;
    [uKS,sKS,vKS] = svd(LapKS);
    f = size(vKS,2);
    kerKS = vKS(:,f-n+1:f);
    for i = 1:N
        kerKS(i,:) = kerKS(i,:) ./ norm(kerKS(i,:));
    end
    svalKS = diag(sKS);
    group3 = kmeans(kerKS,n,'start','sample','maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
end
%
switch SCidx
    case 0
        Grps = [group1,group2,group3];
        SingVals = [svalKU,svalKN,svalKS];
        LapKernel(:,:,1) = kerKU;
        LapKernel(:,:,2) = kerKN;
        LapKernel(:,:,3) = kerKS;
    case 1
        Grps = [group1];
        SingVals = [svalKU];
        LapKernel(:,:,1) = kerKU;
    case 2
        Grps = [group2];
        SingVals = [svalKN];
        LapKernel(:,:,1) = kerKN;
    case 3
        Grps = [group3];
        SingVals = [svalKS];
        LapKernel(:,:,1) = kerKS; 
    otherwise
        display ('parameter error in par.SCidx! It should be specified to 0 or 1 or 2 or 3');  
end

