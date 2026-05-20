close all
clear
clc
ft_path = 'D:\OS(CURRENT)\scripts\2Git\fieldtrip';
if ~exist('ft_defaults','file'), addpath(ft_path); end
ft_defaults;

%% Параметры и данные
% ... (код инициализации данных остается без изменений) ...
elec = load("D:\OS(CURRENT)\data\simulation_support_data\eeg\elec.mat").elec;
G = load('D:\OS(CURRENT)\data\simulation_support_data\eeg\MNE_EEG_FWD_TRPL.mat').MNE_EEG_FWD_TRPL;
NConstSrc = 91; Ntg = 1; flanker = 1; TrLeSe = 5; Fs = 100; NTr = 20; NLclSrc = 10;
Wsize = 1 / 2; Ssize = Wsize / 2;
snr_vec = 10.^(-1.4:0.2:1);
Nsnr = length(snr_vec); NMC = 5;
methods = {'SPoC_\lambda', 'Envelope CorrCA', 'Envelope CorrCA D', 'Envelope CorrCA T'};
nMethods = length(methods);
colors = lines(nMethods);

%% Инициализация массивов метрик
% Результаты: [SNR, MC, Method]
itc_hidden_res  = NaN(Nsnr, NMC, nMethods);
itc_final_res   = NaN(Nsnr, NMC, nMethods);
z_corr_res      = NaN(Nsnr, NMC, nMethods);
cov_corr_res    = NaN(Nsnr, NMC, nMethods);
patt_corr_res   = NaN(Nsnr, NMC, nMethods);
var_hidden_res  = NaN(Nsnr, NMC, nMethods);
var_final_res   = NaN(Nsnr, NMC, nMethods);

fprintf('Запуск симуляции для %d методов...\n', nMethods);
for snr_i = 1:Nsnr
    SNR = snr_vec(snr_i);
    fprintf('Обработка SNR: 10^{%.1f}...\n', log10(SNR));
    
    parfor mc_i = 1:NMC
        [Xtrials, ~, tm, TgPa] = gen_dat_corrca(G, NConstSrc, Ntg, flanker, TrLeSe, Fs, NTr, NLclSrc, SNR);
        tm_epo = epoch_data((tm - min(tm)).^2', Fs, Wsize, Ssize);
        tm_z_true = reshape(mean(tm_epo, 1), [], 1); 
        tm_z_all = repmat(tm_z_true, [NTr, 1]);
        tm_z_all = (tm_z_all - mean(tm_z_all)) ./ std(tm_z_all);

        X_epochs = [];
        for tr_i=1:NTr, X_epochs(:,:,:,tr_i) = epoch_data(Xtrials(:,:,tr_i),Fs,Wsize,Ssize); end
        NEp = size(X_epochs,3);
        X_covs = [];
        for tr_i=1:NTr
            for ep_i=1:NEp, X_covs(:,:,ep_i,tr_i) = cov(squeeze(X_epochs(:,:,ep_i,tr_i))); end
        end

        for m = 1:nMethods
            method_name = methods{m};
            is_spoc = strcmp(method_name, 'SPoC_\lambda');
            
            % 1. Выбор метода и получение весов
            if is_spoc
                [W, A] = spoc(X_epochs(:,:,:), tm_z_all');
                W_cand = W(:,1); 
                A_cand = A(:,1);
            else
                if strcmp(method_name, 'Envelope CorrCA'), [W, A, z_trials] = env_corrca(Xtrials, Fs, Wsize, Ssize);
                elseif strcmp(method_name, 'Envelope CorrCA D'), [W, A, z_trials] = env_corrca_d(Xtrials, Fs, Wsize, Ssize);
                elseif strcmp(method_name, 'Envelope CorrCA T'), [W, A, z_trials] = env_corrca_t(Xtrials, Fs, Wsize, Ssize);
                end
                W_cand = [squeeze(W(1,:,1)); squeeze(W(1,:,end))]';
                A_cand = [squeeze(A(1,:,1)); squeeze(A(1,:,end))]';
                
                % 2. Расчет метрик СКРЫТОЙ компоненты
                z_corr_res(snr_i, mc_i, m) = abs(corr(reshape(z_trials(:, 1, :), [], 1), tm_z_all));
                var_hidden_res(snr_i, mc_i, m) = var(z_trials(:));
                itc_hidden_res(snr_i, mc_i, m) = compute_itc(squeeze(z_trials(:, 1, :)));
            end
            
            % 3. Итоговая компонента (Covariance-based)
            cov_corr_cands = zeros(1, 2);
            for w_i = 1:size(W_cand,2)
                w = W_cand(:, w_i);
                env_vals = zeros(NEp, NTr);
                for tr_i=1:NTr
                    for ep_i = 1:NEp, env_vals(ep_i,tr_i) = w' * squeeze(X_covs(:,:,ep_i,tr_i)) * w; end
                end
                cov_corr_cands(w_i) = abs(corr(env_vals(:), tm_z_all(:)));
            end
            [~, b_idx] = max(cov_corr_cands);
            
            % Финальные расчеты
            final_comp = zeros(NEp, NTr);
            w_best = W_cand(:, b_idx);
            for tr_i = 1:NTr
                for ep_i = 1:NEp, final_comp(ep_i, tr_i) = w_best' * squeeze(X_covs(:,:,ep_i,tr_i)) * w_best; end
            end
            
            cov_corr_res(snr_i, mc_i, m)  = cov_corr_cands(b_idx);
            patt_corr_res(snr_i, mc_i, m) = abs(corr(A_cand(:, b_idx), TgPa));
            itc_final_res(snr_i, mc_i, m) = compute_itc(final_comp);
            var_final_res(snr_i, mc_i, m) = var(final_comp(:));
        end
    end
end

%% Визуализация 1: Корреляции
figure('Name', 'Correlations Analysis', 'Position', [100, 100, 1200, 400]);
metrics1 = {z_corr_res, cov_corr_res, patt_corr_res};
titles1 = {'Hidden Comp Corr', 'Final Comp Corr', 'Pattern Corr'};
ylabels = {'r', 'r', 'r'};
snr_powers = log10(snr_vec);

for k = 1:3
    subplot(1, 3, k); hold on;
    for m = 1:nMethods
        y = squeeze(mean(metrics1{k}(:, :, m), 2))';
        std_dev = squeeze(std(metrics1{k}(:, :, m), 0, 2))';
        
        % Заливка отклонения
        fill([snr_powers, fliplr(snr_powers)], [y-std_dev, fliplr(y+std_dev)], ...
            colors(m,:), 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        plot(snr_powers, y, '-o', 'LineWidth', 1.5, 'Color', colors(m,:), 'DisplayName', methods{m});
    end
    grid on; title(titles1{k}, 'FontWeight', 'bold'); 
    xlabel('log10(SNR)'); ylabel(ylabels{k}); ylim([0 1.05]);
    if k == 1, legend('Location', 'best', 'FontSize', 8); end
end

%% Визуализация 2: ITC и Дисперсия (Финальные)
figure('Name', 'Final Metrics Analysis', 'Position', [100, 100, 800, 400]);
metrics2 = {itc_final_res, var_final_res};
titles2 = {'ITC Final', 'Var Final'};

for k = 1:2
    subplot(1, 2, k); hold on;
    for m = 1:nMethods
        y  = squeeze(mean(metrics2{k}(:, :, m), 2))'; 
        std_dev = squeeze(std(metrics2{k}(:, :, m), 0, 2))';
        
        fill([snr_powers, fliplr(snr_powers)], [y-std_dev, fliplr(y+std_dev)], ...
            colors(m,:), 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        plot(snr_powers, y, '-o', 'LineWidth', 1.5, 'Color', colors(m,:), 'DisplayName', methods{m});
    end
    grid on; title(titles2{k}, 'FontWeight', 'bold'); 
    xlabel('log10(SNR)'); 
    if k == 1, ylabel('ITC'); else, ylabel('Var'); end
    if k == 1, legend('Location', 'best', 'FontSize', 8); end
end

function itc = compute_itc(data_mat)
    n_tr = size(data_mat, 2);
    if n_tr < 2, itc = 0; return; end
    C = corr(data_mat);
    itc = (sum(C(:)) - n_tr) / (n_tr * (n_tr - 1));
end