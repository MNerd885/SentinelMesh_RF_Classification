import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from simulated_movement_visualization import generate_mocap_trajectory

# =====================================================================
# 1. PARAMETRI ACCONEER A121 & CONFIGURAZIONE FINISTRA SCORREVOLE
# =====================================================================
fc = 60e9             # Frequenza portante (60 GHz)
c = 3e8               # Velocità della luce (m/s)
wavelength = c / fc   # Wavelength lambda ~ 5 mm

fs = 1000             # Sweep rate (Slow-time sampling frequency: 1000 Hz)
N_sweeps = 128        # Dimensione della finestra temporale per ogni frame (~128 ms)
frame_step = 30       # Passo di scorrimento tra frame (30 campioni = 30 ms -> ~33 FPS)
ACTIONS = ["walking", "bending", "drinking", "lying", "sitting", "falling"]

# Griglia di distanza (Range Bins)
range_resolution = 0.025
range_bins = np.arange(0.5, 3.5, range_resolution)
N_range_bins = len(range_bins)

# =====================================================================
# 2. GENERAZIONE TRAIETTORIA COMPLETA (3 SECONDI)
# =====================================================================

duration_sec = 3.0
points_3d, rcs_weights, time_vector = generate_mocap_trajectory(ACTIONS[0], duration_sec, fs)
total_samples = len(time_vector)

# =====================================================================
# 3. SETUP FIGURA MATPLOTLIB & ASSE VELOCITÀ
# =====================================================================
N_fft = 256
doppler_freqs = np.fft.fftshift(np.fft.fftfreq(N_fft, d=1/fs))
velocities = (doppler_freqs * wavelength) / 2
window = np.hanning(N_sweeps)

fig, ax = plt.subplots(figsize=(9, 6))

# Matrice fantoccio per inizializzare l'oggetto imshow
dummy_map = np.zeros((N_range_bins, N_fft))
im = ax.imshow(
    dummy_map, 
    aspect='auto', 
    extent=[velocities[0], velocities[-1], range_bins[-1], range_bins[0]],
    cmap='jet',
    vmin=-30, vmax=10
)

ax.set_xlabel('Velocità Radiale (m/s)')
ax.set_ylabel('Distanza Radiale / Range Bin (m)')
ax.invert_yaxis()
ax.grid(True, linestyle='--', alpha=0.3)
fig.colorbar(im, ax=ax, label='Potenza (dB)')
title_text = ax.set_title('Acconeer A121 Range-Doppler Map - t = 0.00 s')

# Numero totale di frame generabili
num_frames = (total_samples - N_sweeps) // frame_step

# =====================================================================
# 4. FUNZIONI PER IL CALCOLO DINAMICO DEL FRAME & ANIMAZIONE
# =====================================================================
def compute_rd_map(start_idx):
    end_idx = start_idx + N_sweeps
    points_frame = points_3d[:, start_idx:end_idx, :]
    
    # Distanza 3D e fase per i 7 punti nella finestra di sweep corrente
    R_k = np.sqrt(points_frame[:, :, 0]**2 + points_frame[:, :, 1]**2 + points_frame[:, :, 2]**2)
    phase_k = (4 * np.pi / wavelength) * R_k
    
    IQ_matrix = np.zeros((N_range_bins, N_sweeps), dtype=complex)
    for k in range(7):
        for n in range(N_sweeps):
            r_target = R_k[k, n]
            sig_phase = phase_k[k, n]
            pulse_shape = np.exp(-((range_bins - r_target) / (2 * range_resolution))**2)
            IQ_matrix[:, n] += rcs_weights[k] * pulse_shape * np.exp(-1j * sig_phase)
            
    # Aggiunta rumore AWGN (SNR = 15 dB)
    SNR_dB = 15
    signal_power = np.mean(np.abs(IQ_matrix)**2) + 1e-12
    noise_power = signal_power / (10**(SNR_dB / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*IQ_matrix.shape) + 1j * np.random.randn(*IQ_matrix.shape))
    
    IQ_noisy = IQ_matrix + noise
    IQ_windowed = IQ_noisy * window[np.newaxis, :]
    
    # Doppler FFT
    rd_map = np.fft.fftshift(np.fft.fft(IQ_windowed, n=N_fft, axis=1), axes=1)
    rd_map_dB = 20 * np.log10(np.abs(rd_map) + 1e-6)
    return rd_map_dB

def update(frame_idx):
    start_idx = frame_idx * frame_step
    current_time = time_vector[start_idx + N_sweeps // 2]
    
    rd_map_dB = compute_rd_map(start_idx)
    
    # Aggiornamento dei dati nell'oggetto grafico (molto più veloce di plt.clf())
    im.set_array(rd_map_dB)
    
    # Normalizzazione dinamica della scala colore ancorata al picco
    max_val = np.max(rd_map_dB)
    im.set_clim(vmin=max_val - 35, vmax=max_val)
    
    title_text.set_text(f'Acconeer A121 Range-Doppler Map ({ACTIONS[0]}) - t = {current_time:.2f} s')
    return [im, title_text]

# Creazione dell'animazione (interval=30 ms rispecchia il tempo reale del frame_step)
anim = FuncAnimation(fig, update, frames=num_frames, interval=30, blit=True)

plt.tight_layout()
plt.show()

anim.save(f"Animations/Range-Doppler maps/{ACTIONS[0]}_rd_map.gif")