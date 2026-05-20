close all
clear
clc

ft_path = 'D:\OS(CURRENT)\scripts\2Git\fieldtrip';
if ~exist('ft_defaults','file')
    addpath(ft_path);
end
ft_defaults;

%%
elec = load("D:\OS(CURRENT)\data\simulation_support_data\eeg\elec.mat").elec;
topo = [];
topo.dimord = 'chan_time';
topo.label  = elec.label;
topo.time   = 0;
topo.elec   = elec;

laycfg = [];
laycfg.elec = elec;
lay = ft_prepare_layout(laycfg);

cfg = [];
cfg.marker       = '';
cfg.layout       = lay;
cfg.comment      = 'no';
cfg.style        = 'fill';
cfg.markersymbol = '';
cfg.colorbar     = 'no'; 
cfg.layout.pos(:, 1:2) = cfg.layout.pos(:, 1:2) * 1.1; 
cfg.layout.pos(:, 2) = cfg.layout.pos(:, 2) - 0.05;

G = load('D:\OS(CURRENT)\data\simulation_support_data\eeg\MNE_EEG_FWD_TRPL.mat').MNE_EEG_FWD_TRPL;

%% 
NConstSrc = 91; Ntg = 1; flanker = 1; TrLeSe = 5; Fs = 100; NTr = 20; NLclSrc = 10;
Wsize = 1 / 2; Ssize = Wsize / 2;
snr_vec = 10.^(-0.5:0.1:1);
Nsnr = length(snr_vec);
NMC = 3;
methods = {'SPoC_\lambda', 'Envelope CorrCA', 'Envelope CorrCA D', 'Envelope CorrCA T'};
nMethods = length(methods);
colors = lines(nMethods);

% Размеры: [SNR, MC, Method]
env_corr_res  = zeros(Nsnr, NMC, nMethods); % Hilbert
cov_corr_res  = zeros(Nsnr, NMC, nMethods); % Covariance-based
patt_corr_res = zeros(Nsnr, NMC, nMethods);
z_corr_res    = zeros(Nsnr, NMC, nMethods);

fprintf('Запуск симуляции для %d методов...\n', nMethods);

for snr_i = 1:Nsnr
    SNR = snr_vec(snr_i);
    fprintf('Обработка SNR: 10^{%.1f}...\n', log10(SNR));
    
    parfor mc_i = 1:NMC
        [Xtrials, Xraw, tm, TgPa] = gen_dat_corrca(G, NConstSrc, Ntg, flanker, TrLeSe, Fs, NTr, NLclSrc, SNR);
        tmraw = repmat(tm,[1,NTr]);
        
        % Подготовка истинного таргета
        tm_epo = epoch_data((tm+min(tm)).^2', Fs, Wsize, Ssize);
        tm_z_true = reshape(mean(tm_epo, 1), [], 1); 
        tm_z_all = repmat(tm_z_true, [NTr, 1]);
        
        for m = 1:nMethods
            method_name = methods{m};
            
            % Выбор метода
            if strcmp(method_name, 'SPoC\lambda')
                [W, A] = spoc(X_epochs, tm_z_all);
            if strcmp(method_name, 'Envelope CorrCA')
                [W, A, z_trials, X_covs] = env_corrca(Xtrials, Fs, Wsize, Ssize);
            elseif strcmp(method_name, 'Envelope CorrCA D')
                [W, A, z_trials, X_covs] = env_corrca_d(Xtrials, Fs, Wsize, Ssize);
            elseif strcmp(method_name, 'Envelope CorrCA T')
                [W, A, z_trials, X_covs] = env_corrca_t(Xtrials, Fs, Wsize, Ssize);
            end
            
            % Берем первый и последний компонент
            W_cand = [squeeze(W(1,:,1)); squeeze(W(1,:,end))]';
            A_cand = [squeeze(A(1,:,1)); squeeze(A(1,:,end))]';
            
            env_corr_cands = zeros(1, 2);
            cov_corr_cands = zeros(1, 2);
            
            for w_i = 1:2
                w = W_cand(:, w_i);
                
                % 1. Hilbert
                env_hilbert = abs(hilbert(w' * Xraw));
                env_corr_cands(w_i) = abs(corr(env_hilbert', tmraw'));
                
                % 2. Covariance-based
                env_cov = zeros(size(X_covs, 3), 1);
                for ep = 1:size(X_covs, 3)
                    env_cov(ep) = w' * X_covs(:,:,ep) * w;
                end
                cov_corr_cands(w_i) = abs(corr(env_cov, tm_z_true));
            end
            
            [~, b_idx] = max(env_corr_cands);
            
            env_corr_res(snr_i, mc_i, m)  = env_corr_cands(b_idx);
            cov_corr_res(snr_i, mc_i, m)  = cov_corr_cands(b_idx);
            patt_corr_res(snr_i, mc_i, m) = abs(corr(A_cand(:, b_idx), TgPa));
            z_corr_res(snr_i, mc_i, m)    = abs(corr(reshape(z_trials(:, 1, :), [], 1), tm_z_all));
        end
    end
end

%% Агрегация и визуализация
snr_powers = log10(snr_vec);

% Порядок в ячейке: {1:Z_corr, 2:Patt_corr, 3:Hilbert_corr, 4:Cov_corr}
metrics = {z_corr_res, patt_corr_res, env_corr_res, cov_corr_res};

% Маппинг для сетки 2x2:
% 1 (TL): Latent, 2 (TR): Covariance
% 3 (BL): Pattern, 4 (BR): Hilbert
plot_map = [1, 4, 2, 3]; 

plot_titles = {'Latent Component Correlation', ...
               'Envelope Correlation (Covariance-based)', ...
               'Spatial Pattern Recovery', ...
               'Envelope Correlation (Hilbert-based)'};

figure('Name', 'SNR Analysis Comparison', 'Position', [100, 100, 1000, 800]);

for k = 1:4
    subplot(2, 2, k); hold on;
    p = plot_map(k); % Индекс метрики для текущей позиции
    
    for m = 1:nMethods
        % Вычисление средних и доверительных интервалов
        y  = squeeze(mean(metrics{p}(:, :, m), 2))'; 
        ci = squeeze(1.96 * std(metrics{p}(:, :, m), 0, 2) / sqrt(NMC))';
        
        % Отрисовка
        plot(snr_powers, y + ci, '--', 'Color', colors(m,:), 'HandleVisibility', 'off');
        plot(snr_powers, y - ci, '--', 'Color', colors(m,:), 'HandleVisibility', 'off');
        plot(snr_powers, y, '-o', 'LineWidth', 1.5, 'Color', colors(m,:), ...
            'MarkerFaceColor', 'w', 'DisplayName', methods{m});
    end
    
    grid on; 
    title(plot_titles{k}, 'FontSize', 12, 'FontWeight', 'bold'); 
    xlabel('signal-to-noise ratio'); 
    ylabel('Correlation (r)');
    ylim([0 1.05]);
    
    % Легенда только для первого графика, чтобы не загромождать остальные
    if k == 1
        legend('Location', 'southeast', 'FontSize', 9); 
    end
end