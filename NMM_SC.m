%% Nonlinear mixture model subspace clustering
% I. Kopriva 2026-07

% NMM_SC

%% Initialization
clear
close all

% Set path to all subfolders
addpath(genpath('.'));

%% Dataset independant parameters

% Number of iterations
nIter = 100;

%% Load the data from the chosen dataset

% Please uncomment the dataset that you want to use and comment the other ones
 dataName = 'YaleBCrop025';
% dataName = 'MNIST';
% dataName = 'USPS';
% dataName = 'ORL';
% dataName = 'COIL20'; 
% dataName = 'COIL100'; % Color images

%% Selection of subspace clustering algorithm

% Please uncomment algorithm you want to use and comment the other ones
% algorithm = 'SSC';  % Sparse subspace clustering
 algorithm = 'GMC_LRSSC';  % GMC constrained low rank sparse SC
% algorithm = 'LRR';  % Nuclear norm low-rank representation SC 
% algorithm = 'S_1o2_LRR'; % L_1/2 norm LRR SC
% algorithm = 'S_2o3_LRR'; % L_2/3 norm LRR SC

%% STEP 1A: prepare data and select hyperparameters of chosen SC algorithm
[paras_data,paras_SC] = params_data_and_algorithms(dataName,algorithm);

i1 = paras_data.i1; i2 = paras_data.i2; % image size
dimSubspace = paras_data.dimSubspace; % Subspace dimension
numIn = paras_data.numIn; % number of in-sample data
nc = paras_data.nc; % number of groups
Y = paras_data.X;  % data
labels = paras_data.labels; % labels

ACC_x_in     = zeros(1, nIter);     NMI_x_in     = zeros(1, nIter);     
Fscore_x_in  = zeros(1, nIter);     Rand_x_in    = zeros(1, nIter);   
Purity_x_in  = zeros(1, nIter);    

for it = 1:nIter
    fprintf('Iter: %d\n',it);

    %% Generate a problem instance
    % Problem instance is a random split of the chosen dataset into an input set (X_in) and an output set (X_out),
    % as well as the concommitant label sets (label_in, label_out)

    rng('shuffle');

    %% STEP 1B prepare in-sample random partitions
    % Each category is separately split, to ensure proportional representation
    nIn = 1; nOut = 1;
    for c=1:nc % Through all categories
        ind = (labels == c); % Indices of the chosen category
        Xc = Y(:,ind);       % Samples ...
        numSamples = size(Xc, 2); % Number of samples ...
        ind = randperm(numSamples); % Random permutation of the indices
        X_in(:,    nIn:nIn+numIn-1 ) = Xc(:, ind(1:numIn)); % Data
        labels_in(  nIn:nIn + numIn-1) = c; % Labels
        nIn  = nIn  + numIn; % Next indices
    end
    X_in( :,   nIn:end) = []; % Cut out the surplus of the allocated space
    labels_in(  nIn:end) = [];

    %% STEP 2: apply selected subspace clustering algorithm(s) to the original data X_in
    [labels_est_X(1,:)] = run_SC_algorithm(X_in,labels_in,paras_SC);

    %% Performance on in-sample data
    ACC_x_in(it)  = 1 - computeCE(labels_est_X,labels_in)     
    NMI_x_in(it) = compute_nmi(labels_in,labels_est_X)       
    Fscore_x_in(it) = compute_f(labels_in,labels_est_X);         
    Rand_x_in(it) = RandIndex(labels_in,labels_est_X);          
    Purity_x_in(it) = purFuc(labels_in,labels_est_X);            

    %% Iterations (END LOOP)
end

display('Estimated performances:')

display('*********** In-sample data:')
mean_ACC_x_in=mean(ACC_x_in)
std_ACC_x_in=std(ACC_x_in)
mean_ACC_wp_in=mean(ACC_wp_in)
std_ACC_wp_in=std(ACC_wp_in)
  
mean_NMI_x_in=mean(NMI_x_in)
std_NMI_x_in=std(NMI_x_in)
mean_NMI_wp_in=mean(NMI_wp_in)
std_NMI_wp_in=std(NMI_wp_in)

mean_Fscore_x_in=mean(Fscore_x_in)
std_Fscore_x_in=std(Fscore_x_in)
mean_Fscore_wp_in=mean(Fscore_wp_in)
std_Fscore_wp_in=std(Fscore_wp_in)

mean_Rand_x_in=mean(Rand_x_in)
std_Rand_x_in=std(Rand_x_in)
mean_Rand_wp_in=mean(Rand_wp_in)
std_Rand_wp_in=std(Rand_wp_in)

mean_purity_x_in=mean(Purity_x_in)
std_purity_x_in=std(Purity_x_in)
mean_purity_wp_in=mean(Purity_wp_in)
std_purity_wp_in=std(Purity_wp_in)

mean_affinity_x = mean(affinity_x)
std_affinity_x = std(affinity_x)
mean_affinity_wp = mean(affinity_wp)
std_affinity_wp = std(affinity_wp)

