function C_ref = log_euclidean_mean(Epochs_cov)
    % Epochs_cov: [n_chan x n_chan x n_epochs]
    [n_chan, ~, n_epochs] = size(Epochs_cov);
    
    sum_log = zeros(n_chan, n_chan);
    for i = 1:n_epochs
        % Обязательно симметризуем перед logm для стабильности
        C = (Epochs_cov(:,:,i) + Epochs_cov(:,:,i)') / 2;
        % Небольшая регуляризация, чтобы logm не выдавал комплексные числа
        C = C + 1e-6 * eye(n_chan); 
        sum_log = sum_log + logm(C);
    end
    
    % Усредняем логарифмы и возвращаем на многообразие через expm
    mean_log = sum_log / n_epochs;
    C_ref = expm((mean_log + mean_log') / 2);
end