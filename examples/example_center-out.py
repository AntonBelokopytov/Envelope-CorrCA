# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 19:29:56 2026

@author: ansbel
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# %%
# For parkinson example chose 8-12 Hz filter below
# For control example chose 15-25 Hz filter below
dpath = 'data/Patient1_ConditionOff_CenterOut_2-35Hz_clear_epochs.fif'
# dpath = 'Control_4_CenterOut_epochs.fif'

epochs = mne.read_epochs(dpath, preload=True)

# %%
epochs_eeg = epochs.copy().pick_channels(epochs.ch_names[:38])
epochs_vars = epochs.copy().pick_channels(epochs.ch_names[38:])

# %%
# 1. Извлекаем монтаж
montage = epochs.copy().get_montage()
pos = montage.get_positions()['ch_pos']

# 2. Смещаем все координаты на 2 см назад по оси Y
for ch in pos:
    pos[ch][1] -= 0.02  # Сдвиг по Y на -20мм

# 3. Пересоздаем монтаж с новыми координатами
new_montage = mne.channels.make_dig_montage(ch_pos=pos, coord_frame='head')

# 4. Применяем его к данным
epochs_eeg.set_montage(new_montage)

# Теперь рисуется корректно по умолчанию
epochs_eeg.plot_sensors(show_names=True)

# %%
mask = (
    (epochs.metadata['pp'] == 2) &
    (epochs.metadata['correct_trials'] == 1)
)

idx = np.where(mask)[0]
idx

# %%
epochs_pp2 = epochs[idx]

# %%
X = epochs_pp2.get_data()[:,:38]
Fs = epochs_pp2.info['sfreq']
fmin = 8
fmax = 12

from scipy.signal import butter, filtfilt
band = [fmin, fmax]
b, a = butter(4, np.array(band) / (Fs / 2), btype='band')
Xfilt = filtfilt(b, a, X, axis=-1)

import src.env_corrca as env_functions
W, A, corrs, Epochs_cov, eigenvalues = env_functions.env_corrca(Xfilt,Fs)

# %%
# Check eigenvalues and correlations!
# Opposite components might matter: according to the
# absolute value of eigenvalue OR InterTrial correlation
plt.figure()

plt.stem(
    eigenvalues,
    linefmt='b-',
    markerfmt='bo',
    basefmt=" "
)

plt.stem(
    corrs,
    linefmt='r-',
    markerfmt='ro',
    basefmt=" "
)

plt.xlabel('num of component')
plt.legend([
    'Eigenvalue',
    'InterTrial correlation'
])

plt.show()

# %%
com_idx = 37
w = W[:,com_idx]
a = A[:,com_idx]

# Лучше сделать максимальное абсолютное значение паттерна положительным
imax = np.argmax(np.abs(a))
sign_flip = np.sign(a[imax])
a = a * sign_flip

fig, axes = plt.subplots(1, 2, figsize=(8, 4), facecolor='white')

# -------- Filter --------
mne.viz.plot_topomap(
    w,
    epochs_eeg.info,
    axes=axes[0],
    names=epochs_eeg.ch_names,   
    show=True,
    cmap='RdBu_r',
)

axes[0].set_title('Filter')

# -------- Pattern --------
mne.viz.plot_topomap(
    a,
    epochs_eeg.info,
    axes=axes[1],
    names=epochs_eeg.ch_names,   
    show=True,
    cmap='RdBu_r',
)

axes[1].set_title('Pattern')

plt.tight_layout()
plt.show()


# %% GET ANALYTIC SIGNAL
from scipy.signal import hilbert
com_idx = 37
w = W[:,com_idx]
a = A[:,com_idx]

# Лучше сделать максимальное абсолютное значение паттерна положительным
imax = np.argmax(np.abs(a))
sign_flip = np.sign(a[imax])
a = a * sign_flip

Xcomp = w @ np.permute_dims(Xfilt,(1,0,2)).reshape([38,-1])
Xcomp = Xcomp.reshape([X.shape[0],X.shape[2]])

analytic = hilbert(Xcomp, axis=0)
envelope = np.abs(analytic)

# %% 
# Keep in mind that the graphs show edge effects.
fig, axes = plt.subplots(1, 2, figsize=(8, 4), facecolor='white')

# -------- Filter --------
mne.viz.plot_topomap(
    w,
    epochs_eeg.info,
    axes=axes[0],
    names=epochs_eeg.ch_names,   
    show=True,
    cmap='RdBu_r',
)

axes[0].set_title('Filter')

# -------- Pattern --------
mne.viz.plot_topomap(
    a,
    epochs_eeg.info,
    axes=axes[1],
    names=epochs_eeg.ch_names,   
    show=True,
    cmap='RdBu_r',
)

axes[1].set_title('Pattern')

plt.tight_layout()
plt.show()


n_epochs, T = envelope.shape

# --- время в секундах ---
time_sec = np.arange(T) / Fs
t_min = time_sec[0]
t_max = time_sec[-1]

mean_envelope = np.mean(envelope, axis=0)
std_envelope  = np.std(envelope, axis=0)

# ---- можно задать верхний предел яркости ----
vmax = 3 * np.max(std_envelope)   # ← меняй при необходимости
vmin = 0

fig = plt.figure(figsize=(10, 8), facecolor='white')
gs = GridSpec(3, 2, width_ratios=[20, 1],
              height_ratios=[2, 1, 1], figure=fig)

# =========================
# 1 Heatmap
# =========================
ax1 = fig.add_subplot(gs[0, 0])
cax = fig.add_subplot(gs[0, 1])

im = ax1.imshow(
    envelope,
    aspect='auto',
    origin='lower',
    extent=[t_min, t_max, 0, n_epochs],
    cmap='viridis',
    vmin=vmin,
    vmax=vmax
)

fig.colorbar(im, cax=cax)

ax1.set_ylabel('Epoch')
ax1.set_title('Envelope per epoch')

# =========================
# 2 All epochs (raw)
# =========================
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

ax2.plot(time_sec, Xcomp.T, alpha=0.5)
ax2.set_ylabel('Amplitude')
ax2.set_title('Component activity (all epochs)')
ax2.grid(True)

# =========================
# 3 Mean envelope ± SD
# =========================
ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

ax3.fill_between(
    time_sec,
    mean_envelope - std_envelope,
    mean_envelope + std_envelope,
    alpha=0.2
)

ax3.plot(time_sec, mean_envelope, linewidth=2)

ax3.set_ylabel('Envelope')
ax3.set_xlabel('Time (s)')
ax3.set_title('Mean envelope ± SD')
ax3.grid(True)

plt.tight_layout()
plt.show()