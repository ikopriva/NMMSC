function [labels_est_X] = run_SC_algorithm(X,labels,paras)

% Runs selected subspace clustering algorithm on data in amninet space (X) 
%
%  Inputs:
%

algorithm = paras.algorithm;
nc=max(labels);   % number of clusters

if strcmp (algorithm,'SSC')
    fprintf('Running SSC ...\n');

    outlier = paras.outlierAmbient; affine = paras.affineAmbient;  r= 0; rho = 2.0;
    ipd_ambient_domain=paras.ipd_ambient_domain;
    alpha_X = paras.alphaAmbient;
    [Z_x,labels_est_X] = SSC(normc(X),r,affine,alpha_X,outlier,rho,labels);
    if ipd_ambient_domain
        d_x = paras.dimSubspaceAmbient;
        C_sym = BuildAdjacency_cut(Z_x,d_x);
        labels_est_X = SpectralClusteringL(C_sym,nc);
    else
        labels_est_X = SpectralClusteringL(abs(Z_x)+abs(Z_x'),nc);
    end

elseif strcmp(algorithm,'GMC_LRSSC')
    fprintf('Running GMC_LRSSC ...\n');

    alpha_x = paras.alphaAmbient; lambda_x = paras.lambdaAmbient;
    gamma_x = paras.gammaAmbient;

    ipd_ambient_domain=paras.ipd_ambient_domain;

    options = struct('lambda',lambda_x,'alpha',alpha_x,'rank_est',0.6,'gamma',gamma_x,...
        'err_thr',1e-4,'iter_max',100, 'affine',false,...
        'l1_nucl_norm',false,'l0norm',false,'elra',false, 'gmc',true);
    %
    [Z_x, error] = ADMM_LRSSC(normc(X),options);
    if ipd_ambient_domain
        d_x = paras.dimSubspaceAmbient;
        C_sym = BuildAdjacency_cut(abs(Z_x),d_x);
        labels_est_X = SpectralClusteringL(C_sym,nc);
    else
        labels_est_X = SpectralClusteringL(abs(Z_x)+abs(Z_x'),nc);
    end

elseif strcmp(algorithm,'LRR')
    fprintf('Running LRR..\n');

    ipd_ambient_domain=paras.ipd_ambient_domain;

    lambda_x = paras.lambdaAmbient;
    Z_x = solve_lrr(normc(X),lambda_x);
    if ipd_ambient_domain
        d_x = paras.dimSubspaceAmbient;
        Z_x = BuildAdjacency_cut(Z_x,d_x); % keep largest d_x coefficients
    end

    % post processing
    [U,S,V] = svd(Z_x,'econ');
    S = diag(S);
    r = sum(S>1e-4*S(1));
    U = U(:,1:r);S = S(1:r);
    U = U*diag(sqrt(S));
    U = normr(U);
    U = U./repmat(sqrt(sum(U.^2,2)),1,size(U,2));
    LL = (U*U').^4;

    % spectral clustering
    D = diag(1./sqrt(sum(LL,2)));
    LL = D*LL*D;

    [U,S,V] = svd(LL);
    V = U(:,1:nc);
    V = D*V;
    labels_est_X = kmeans(V,nc,'emptyaction','singleton','replicates',10,'display','off');
elseif strcmp(algorithm,'S_1o2_LRR')
    fprintf('Running LRR_S_1/2..\n');

    ipd_ambient_domain=paras.ipd_ambient_domain;

    lambda_x = paras.lambdaAmbient;
    Z_x = Run_SpNM('spdual12lrr',normc(X),lambda_x);
    if ipd_ambient_domain
        d_x = paras.dimSubspaceAmbient;
        C_sym = BuildAdjacency_cut(Z_x,d_x);
        labels_est_X = SpectralClusteringL(C_sym,nc);
    else
        labels_est_X = SpectralClusteringL(abs(Z_x)+abs(Z_x'),nc);
    end

elseif strcmp(algorithm,'S_2o3_LRR')
    fprintf('Running LRR_S_2/3..\n');
%     
    ipd_ambient_domain=paras.ipd_ambient_domain;

   lambda_x = paras.lambdaAmbient;
   Z_x = Run_SpNM('spdual23lrr',normc(X),lambda_x); 
    if ipd_ambient_domain
        d_x = paras.dimSubspaceAmbient;
        C_sym = BuildAdjacency_cut(Z_x,d_x);
        labels_est_X = SpectralClusteringL(C_sym,nc);
    else
        labels_est_X = SpectralClusteringL(abs(Z_x)+abs(Z_x'),nc);
    end
end

end
