function [W, A, Z_epochs, X_epochs] = env_corrca_ht(X, Fs, Wsize, Ssize, lambda)
    if nargin < 5
        lambda = 1e-5; 
    end
    [T, D, n_trials] = size(X);
    
    % 1. Вычитание ERP
    for tr_idx=1:n_trials
        X(:,:,tr_idx) = X(:,:,tr_idx) - mean(X(:,:,tr_idx),1);
    end
    Xmean = mean(X,3);
    
    % Нормировка исходных сигналов
    for tr_idx=1:n_trials
        mX = X(:,:,tr_idx) - Xmean;
        mX = mX ./ sqrt(trace(cov(mX)));
        X(:,:,tr_idx) = mX;
    end
    
    % 2. Преобразование Гильберта и обрезка краев
    Xh_full = zeros(T, D, n_trials);
    for tr_idx=1:n_trials
        Xh_full(:,:,tr_idx) = hilbert(X(:,:,tr_idx));
    end
    
    pad_len = round(Fs/4);
    Xh = Xh_full(pad_len+1 : end-pad_len, :, :);
    
    % 3. Эпохирование комплексного (аналитического) и вещественного сигналов
    temp_epo = epoch_data(Xh(:,:,1), Fs, Wsize, Ssize);
    [W_samples, ~, n_epochs] = size(temp_epo);
    
    Xh_epochs = zeros(W_samples, D, n_epochs, n_trials);
    X_epochs = zeros(W_samples, D, n_epochs, n_trials);
    
    for tr_idx=1:n_trials
        Xh_epochs(:,:,:,tr_idx) = epoch_data(Xh(:,:,tr_idx), Fs, Wsize, Ssize);
        X_epochs(:,:,:,tr_idx) = epoch_data(X(pad_len+1:end-pad_len,:,tr_idx), Fs, Wsize, Ssize);
    end
    
    % 4. Расчет комплексных эрмитовых ковариационных матриц по эпохам
    X_covs_h = zeros(D, D, n_epochs, n_trials);
    for j = 1:n_trials
        for i = 1:n_epochs
            X_segment = squeeze(Xh_epochs(:,:,i,j)); 
            X_covs_h(:,:,i,j) = (X_segment' * X_segment) / (W_samples - 1);
        end
    end
    
    % 5. Переход в комплексное Касательное Пространство (Tangent Space)
    % Схлопываем эпохи и триалы в один 3D-тензор [D x D x (n_epochs * n_trials)]
    COV_3D = reshape(X_covs_h, D, D, n_epochs * n_trials);
    
    % Вызов вашей целевой римановой функции (с поддержкой комплексных матриц)
    [Feat_all, ~] = Tangent_space_complex(COV_3D, lambda);
    
    % Возвращаем трехмерную структуру [n_epochs x K_tangent x n_trials] для CorrCA
    % К размерности K_tangent теперь равна D*D из-за мнимых частей
    K_tangent = D * D; 
    X_tangent_vec = zeros(n_epochs, K_tangent, n_trials);
    for j = 1:n_trials
        % Вырезаем признаки для конкретного триала
        start_idx = (j-1)*n_epochs + 1;
        end_idx = j*n_epochs;
        X_tangent_vec(:,:,j) = Feat_all(:, start_idx:end_idx)';
    end
        
    % 6. Применение классического вещественного CorrCA к касательному пространству
    [Vc, ~, ~] = corrca(X_tangent_vec, lambda);
    
    % Коррекция неопределенности знака для компонент CorrCA (метод макс. элемента)
    for c = 1:size(Vc, 2)
        [~, max_idx] = max(abs(Vc(:, c)));
        if Vc(max_idx, c) < 0
            Vc(:, c) = -Vc(:, c);
        end
    end
    
    % 7. Извлечение проекций компонент (динамика по эпохам)
    n_comps = size(Vc, 2);
    Z_epochs = zeros(n_epochs, n_comps, n_trials);
    
    for j = 1:n_trials
        trial_tangent = X_tangent_vec(:,:,j);
        z_tr = trial_tangent * Vc; 
        Z_epochs(:,:,j) = (z_tr - mean(z_tr, 1)) ./ std(z_tr, [], 1);
    end
    
    % 8. Расчет стандартных ковариаций вещественного сигнала для SPoC
    [~, n_channels, ~, ~] = size(X_epochs);
    X_covs = zeros(n_channels, n_channels, n_epochs, n_trials);
    for j=1:n_trials
        for i=1:n_epochs
            X_covs(:,:,i,j) = cov(X_epochs(:,:,i,j));
        end
    end

    % 9. Расчет пространственных фильтров SPoC на основе профилей из Tangent Space
    n_comps_to_calc = min(10, n_comps);
    W = zeros(n_comps_to_calc, n_channels, n_channels);
    A = zeros(n_comps_to_calc, n_channels, n_channels);
    
    for comp_i = 1:n_comps_to_calc
        z_trials_comp = squeeze(Z_epochs(:, comp_i, :));
        [w, a] = my_spoc(X_covs, z_trials_comp, lambda);
        
        % Коррекция знака пространственных паттернов SPoC для топокарт
        for src_j = 1:size(w, 2)
            [~, max_ch] = max(abs(a(:, src_j))); 
            if a(max_ch, src_j) < 0
                w(:, src_j) = -w(:, src_j);
                a(:, src_j) = -a(:, src_j);
            end
        end
        
        W(comp_i,:,:) = w;
        A(comp_i,:,:) = a;
    end
end

% =========================================================================
% ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
% =========================================================================

function [Feat, C] = Tangent_space_complex(COV, lambda)
    [N_elec, ~, NTrial] = size(COV);
    Feat = zeros(N_elec * N_elec, NTrial);

    C = mean(COV, 3);
    C = C + lambda * eye(N_elec) * trace(C) / N_elec;
    
    [U, S] = eig(C);
    s = max(diag(S), 1e-10 * max(diag(S))); 
    Pinv_sqrt = U * diag(1 ./ sqrt(s)) * U';

    for i = 1:NTrial
        Tn = logm(Pinv_sqrt * COV(:,:,i) * Pinv_sqrt);
        
        diag_part = real(diag(Tn));
        
        mask_upper = triu(true(N_elec), 1);
        upper_elements = Tn(mask_upper);
        
        % Разделяем комплексные числа на Re и Im с весом sqrt(2)
        re_part = real(upper_elements) * sqrt(2);
        im_part = imag(upper_elements) * sqrt(2);
        
        Feat(:, i) = [diag_part; re_part; im_part];
    end
end

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
