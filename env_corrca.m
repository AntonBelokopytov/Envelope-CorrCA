function [W, A, corrs, Epochs_cov, S] = env_corrca(X, ...
Fs, Wsize, Ssize, method)

opt.X_min_var_explained = 0.99;
opt.whitening_reg = 0.00001; % 0.00001

[~, n_ch, n_trials] = size(X);

X_epochs = [];
for tr_idx=1:n_trials
    X(:,:,tr_idx) = X(:,:,tr_idx) - mean(X(:,:,tr_idx),1);
    X(:,:,tr_idx) = X(:,:,tr_idx) ./ sqrt(trace(cov(X(:,:,tr_idx))));
    X_epochs(:,:,:,tr_idx) = epoch_data(X(:,:,tr_idx),Fs,Wsize,Ssize);
end

[~, ~, n_epochs, ~] = size(X_epochs);

[VecCov, Wm, Cxx, Epochs_cov, Epochs_covW] = get_white_covariance_series(X_epochs(:,:,:), opt);
Epochs_cov = reshape(Epochs_cov, [n_ch, n_ch, n_epochs, n_trials]);
Epochs_covW = reshape(Epochs_covW, [n_ch, n_ch, n_epochs, n_trials]);

if strcmp(method, 'light')
    [W, A, S] = eig_dec(Epochs_covW, Wm, Cxx);

elseif strcmp(method, 'full')
    [VecCovdr, Uf] = project_to_pc(VecCov, opt.X_min_var_explained);
    [n_feat,~] = size(VecCovdr);
    
    VecCovdr = reshape(VecCovdr, [n_feat, n_epochs, n_trials]);
    VecCovdr = permute(VecCovdr, [2,1,3]);

    [Vf, Rw, Rb] = corrca(VecCovdr);
    Af =  Uf * Rw * Vf(:,1);

    [W, A, S] = project_filters_to_manifold(Af, Wm, Cxx);
end

[~,n_components] = size(W);
for comp_idx=1:n_components
    Envs=[];
    for ep_idx=1:n_epochs
        for tr_idx=1:n_trials
            w = squeeze(W(:,comp_idx));
            Envs(ep_idx,tr_idx) = w'*Epochs_cov(:,:,ep_idx,tr_idx)*w;
        end
    end
    inters_c = corr(Envs);
    corr_mask = triu(true(size(inters_c)),1);
    corrs(comp_idx) = mean(inters_c(corr_mask));
end

figure;
stem(S, 'filled'); 
hold on
stem(corrs, 'filled');

xlabel('num of component')

legend({'Eigenvalue', ...
        'InterTrial correlation'}, ...
        'Interpreter','latex')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function X_epo = epoch_data(X, Fs, Ws, Ss)

W = fix(Ws*Fs);
S = fix(Ss*Fs);
range = 1:W; ep = 1;
X_epo = [];
while range(end) <= size(X,1)
    X_epo(:,:,ep) = X(range,:); 
    range = range + S; ep = ep + 1;
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [v] = cov2upper(C)
    upper_triu_mask = triu(true(size(C)),1);
    upper_mask = triu(true(size(C)));
    C(upper_triu_mask) = C(upper_triu_mask)*sqrt(2);
    upper_triangle = C(upper_mask);
    v = upper_triangle(:);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function C = upper2cov(v)
    n = (-1 + sqrt(1 + 8 * numel(v))) / 2;
    assert(mod(n,1) == 0, 'Vector length does not correspond to a triangular matrix.');

    C = zeros(n);
    upper_mask = triu(true(n));
    C(upper_mask) = v;

    upper_triu_mask = triu(true(n), 1);
    C(upper_triu_mask) = C(upper_triu_mask) / sqrt(2);

    C = C + triu(C, 1)';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [VecCov, Wm, Cxx, Epochs_cov, Epochs_covW] = get_white_covariance_series(X_epochs, opt)

% Function to get upper triangular covarience time series in dimension
% reduced space

[~,n_channels,n_epochs] = size(X_epochs);
n_features = (n_channels^2-n_channels)/2+n_channels;

Epochs_cov = zeros(n_channels,n_channels,n_epochs);
for ep_idx = 1:n_epochs
    Xcov = cov(X_epochs(:,:,ep_idx));
    Epochs_cov(:,:,ep_idx) = Xcov;
end
% Mean covariance matrix
Cxx = mean(Epochs_cov,3);

% Whitening matrix
Cxx_r = Cxx+opt.whitening_reg*eye(size(Cxx))*trace(Cxx)/size(Cxx,1);
iWm = sqrtm(Cxx_r);    
Wm = eye(n_channels) / iWm;

% Whightened covariance series (upper triangular parts)
Epochs_covW = zeros(n_channels,n_channels,n_epochs);
VecCov = zeros(n_features,n_epochs);
for ep_idx = 1:n_epochs
    XcovW = Wm * Epochs_cov(:,:,ep_idx) * Wm';
    Epochs_covW(:,:,ep_idx) = XcovW;
    VecCov(:, ep_idx) = cov2upper(XcovW);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [X, U] = project_to_pc(X, min_var_explained)

if min_var_explained ~= 1 % Dim reduction
    % PCA & dimension reduction ( svd() is faster than eig(cov(X)) )
    [U,S,~] = svd(X,"econ");

    S = diag(S);
    % compute an estimate of the rank of the data
    tol = max(size(X)) * eps(S(1));
    r = sum(S > tol);
    % compute cumulative variance explained
    ve = S.^2;
    var_explained = cumsum(ve) / sum(ve);
    var_explained(end) = 1;
    n_components = find(var_explained>=min_var_explained, 1);
    n_components = max(min(n_components, r), 1);
    U = U(:,1:n_components);

    X = U'*X;
else
    U = eye(size(X,1));
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [W, Rw, Rb] = corrca(X)

gamma = 0.00001; % shrinkage parameter

[T, D, N] = size(X);

Xc = X - mean(X,1);

% инициализация
Rw = zeros(D,D);
Rb = zeros(D,D);

for i = 1:N
    
    Xi = squeeze(Xc(:,:,i)); % T × D
    
    % within-subject covariance
    Ci = (Xi' * Xi) / (T-1);
    Rw = Rw + Ci;
    
    % between-subject covariance
    for j = i+1:N
        
        Xj = squeeze(Xc(:,:,j));
        
        Cij = (Xi' * Xj) / (T-1);
        Rb = Rb + Cij;
    end
end

% нормировка
Rw = Rw / N;
Rb = (Rb + Rb') / (N*(N-1));

% shrinkage regularization
Rw_reg = (1-gamma)*Rw + gamma*mean(eig(Rw))*eye(D);

% generalized eigenvalue problem
[W, S] = eig(Rb, Rw_reg, 'chol');

% сортировка
[~, indx] = sort(diag(S), 'descend');
W = W(:, indx);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [W, A, S] = eig_dec(Epochs_covW, Wm, Cxx)

gamma = 0.001;

[~, n_channels, n_epochs, n_trials] = size(Epochs_covW);

for sub_idx=1:n_trials
    tr = trace(mean(Epochs_covW(:,:,:,sub_idx),3));
    for i=1:n_epochs
        Ci = Epochs_covW(:,:,i,sub_idx);
        Epochs_covW(:,:,i,sub_idx) = Ci / tr;
    end

    m = mean(Epochs_covW(:,:,:,sub_idx),3);
    for i=1:n_epochs
        Ci = Epochs_covW(:,:,i,sub_idx);
        Epochs_covW(:,:,i,sub_idx) = Ci - m;
    end
end

Rw = zeros(n_channels,n_channels);
Rb = zeros(n_channels,n_channels);
for ep_idx=1:n_epochs
    for i=1:n_trials
        Ci = Epochs_covW(:,:,ep_idx,i);
        Rw = Rw+Ci*Ci;
        for j=i+1:n_trials
            Cj = Epochs_covW(:,:,ep_idx,j); 
            Rb = Rb+Ci*Cj;
        end
    end
end
Rw = (Rw + Rw') / 2;
Rb = (Rb + Rb') / 2;

Rw = Rw/n_trials/n_epochs;
Rb = Rb/(n_trials*(n_trials-1))/n_epochs;

Rw = (1-gamma)*Rw + gamma*mean(eig(Rw))*eye(size(Rw));

[w,S]=eig(Rb,Rw); [S,indx]=sort(diag(S),'descend'); w=w(:,indx);
Wpr = Wm*w;

for comp_idx=1:size(Wpr,2)
    Wprn = Wpr(:,comp_idx) / sqrt(Wpr(:,comp_idx)' * Cxx * Wpr(:,comp_idx));
    W(:,comp_idx) = Wprn;
    A(:,comp_idx) = Cxx * Wprn / (Wprn' * Cxx * Wprn);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [W, A, S] = project_filters_to_manifold(V, Wm, Cxx)

% Project filters to manifold
WW = upper2cov(V);

[Uw,S] = eig(WW);S=diag(S);[S,idxs]=sort(S,'descend');Uw=Uw(:,idxs);
% [Uw,S,~] = svd(WW);s=diag(S);
% stem(s)
% xlabel('number of component')
% ylabel('\lambda value')
% title('Spectrum of eigenvalues of the matrix W')
% Optionally svd() could be used instead of eig() (Result is the same. Order differs)
% [Uw,~,~] = svd(WW);

% Normalization and pattern recovery
for local_src_idx=1:size(Uw,2)
    % Return filters from the whightened space
    wi = Wm * Uw(:,local_src_idx);
    % Normalize
    Wprn = wi / sqrt(wi' * Cxx * wi);
    W(:,local_src_idx) = Wprn;
    A(:,local_src_idx) = Cxx * Wprn / (Wprn' * Cxx * Wprn);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
