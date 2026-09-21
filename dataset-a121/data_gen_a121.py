
import numpy as np
import scipy.signal as signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# =====================================================================
# 1. Physical and electromagnetic paremeters for the Acconeer A121
# =====================================================================
FC = 60.5e9               # Carrier frequency: 60.5 GHz
C = 3.0e8                 # Light speed (m/s)
WAVELENGTH = C / FC       # Wavelenght (~4.95 mm)
SWEEP_RATE = 500          # Sweep frequency (Hz) - slow-time
NUM_SWEEPS_PER_FRAME = 64 # Number of sweeps for a single Range-Doppler map
RANGE_RESOLUTION = 0.025  # Resolution/distance step (2.5 cm)
NUM_RANGE_BINS = 40       # Number of distance bins
START_RANGE = 0.5         # Minimal measured distance (m)
NOISE_STD = 0.05          # Std deviation of thermal noise/AWGN phase

ACTIONS = ["walking", "bending", "drinking", "lying", "sitting", "falling"]

# =====================================================================
# 2. SIMULATORE CINEMATICO (STILE CMU MOCAP DATASET)
# =====================================================================
def generate_mocap_trajectory(action, duration_sec=2.0, fs=SWEEP_RATE):
    """
    Simula le coordinate 3D (x, y, z) nel tempo di 5 scatterer corporei principali:
    0: Torso (RCS = 1.0)
    1: Testa (RCS = 0.3)
    2: Mano Destra (RCS = 0.15)
    3: Mano Sinistra (RCS = 0.15)
    4: Gamba/Piede Sinistro (RCS = 0.2)
    5: Gamba/Piede Destro (RCS = 0.2)
    """
    num_samples = int(duration_sec * fs)
    t = np.linspace(0, duration_sec, num_samples)
    
    # Inizializzazione posizioni (x: laterale, y: distanza dal radar, z: altezza)
    # Radar situato all'origine (0, 0, 1.0)
    points = np.zeros((6, num_samples, 3))
    rcs = np.array([1.0, 0.3, 0.15, 0.15, 0.2, 0.2])
    
    # Generazione traiettorie in base alla classe
    if action == "walking":
        y_base = 2.5 - 0.6 * t # Avanzamento verso il radar
        points[0] = np.column_stack([np.zeros_like(t), y_base, np.full_like(t, 1.2)]) # Torso
        points[1] = np.column_stack([np.zeros_like(t), y_base, np.full_like(t, 1.6)]) # Testa
        
        # Mani in controfase (sfasamento di pi radianti tra destra e sinistra)
        points[2] = np.column_stack([0.3 * np.sin(2*np.pi*2*t), y_base, 1.0 + 0.2*np.cos(2*np.pi*2*t)])
        points[3] = np.column_stack([-0.3 * np.sin(2*np.pi*2*t + np.pi), y_base, 1.0 + 0.2*np.cos(2*np.pi*2*t + np.pi)])
        
        # Gambe/Piedi in controfase
        points[4] = np.column_stack([-0.2 * np.ones_like(t), y_base + 0.2*np.sin(2*np.pi*1.5*t), 0.3 + 0.2*np.sin(2*np.pi*1.5*t)])
        points[5] = np.column_stack([0.2 * np.ones_like(t), y_base - 0.2*np.sin(2*np.pi*1.5*t), 0.3 - 0.2*np.sin(2*np.pi*1.5*t)])
        
    elif action == "bending":
        y_base = 1.8
        bend_factor = np.sin(np.pi * t / duration_sec) # Flessione in avanti
        points[0] = np.column_stack([np.zeros_like(t), y_base - 0.3 * bend_factor, 1.2 - 0.4 * bend_factor])
        points[1] = np.column_stack([np.zeros_like(t), y_base - 0.5 * bend_factor, 1.6 - 0.7 * bend_factor])
        points[2] = np.column_stack([0.2 * bend_factor, y_base - 0.6 * bend_factor, 1.0 - 0.8 * bend_factor])  # Mano DX a terra
        points[3] = np.column_stack([-0.2 * bend_factor, y_base - 0.6 * bend_factor, 1.0 - 0.8 * bend_factor]) # Mano SX a terra
        points[4] = np.column_stack([-0.2 * np.ones_like(t), np.full_like(t, y_base), np.full_like(t, 0.2)])
        points[5] = np.column_stack([0.2 * np.ones_like(t), np.full_like(t, y_base), np.full_like(t, 0.2)])

    elif action == "drinking":
        y_base = 1.5
        drink_hand = np.sin(np.pi * t / duration_sec)
        points[0] = np.column_stack([np.zeros_like(t), np.full_like(t, y_base), np.full_like(t, 1.2)])
        points[1] = np.column_stack([np.zeros_like(t), np.full_like(t, y_base), np.full_like(t, 1.6)])
        points[2] = np.column_stack([np.zeros_like(t), y_base - 0.2 * drink_hand, 0.8 + 0.7 * drink_hand]) # Mano DX al viso
        points[3] = np.column_stack([-0.3 * np.ones_like(t), np.full_like(t, y_base), np.full_like(t, 0.8)])# Mano SX ferma
        points[4] = np.column_stack([-0.2 * np.ones_like(t), np.full_like(t, y_base), np.full_like(t, 0.2)])
        points[5] = np.column_stack([0.2 * np.ones_like(t), np.full_like(t, y_base), np.full_like(t, 0.2)])

    elif action == "lying":
        y_base = 2.0
        micro_breath = 0.003 * np.sin(2 * np.pi * 0.3 * t)
        
        # Torso (con micro-respirazione)
        points[0] = np.column_stack([np.zeros_like(t), y_base + micro_breath, np.full_like(t, 0.4)])
        # Testa
        points[1] = np.column_stack([np.zeros_like(t), np.full_like(t, y_base + 0.4), np.full_like(t, 0.4)])
        # Mano DX
        points[2] = np.column_stack([0.3 * np.ones_like(t), np.full_like(t, y_base), np.full_like(t, 0.4)])
        # Mano SX
        points[3] = np.column_stack([-0.3 * np.ones_like(t), np.full_like(t, y_base), np.full_like(t, 0.4)])
        # Gamba SX
        points[4] = np.column_stack([-0.2 * np.ones_like(t), np.full_like(t, y_base - 0.5), np.full_like(t, 0.4)])
        # Gamba DX
        points[5] = np.column_stack([0.2 * np.ones_like(t), np.full_like(t, y_base - 0.5), np.full_like(t, 0.4)])
    
    elif action == "sitting":
        y_base = 1.7
        sit_progress = np.clip(t / (duration_sec * 0.7), 0, 1)
        points[0] = np.column_stack([np.zeros_like(t), y_base + 0.1*sit_progress, 1.2 - 0.6 * sit_progress])
        points[1] = np.column_stack([np.zeros_like(t), y_base + 0.1*sit_progress, 1.6 - 0.6 * sit_progress])
        points[2] = np.column_stack([0.3 * np.ones_like(t), np.full_like(t, y_base), 0.9 - 0.5 * sit_progress])
        points[3] = np.column_stack([-0.3 * np.ones_like(t), np.full_like(t, y_base), 0.9 - 0.5 * sit_progress])
        points[4] = np.column_stack([-0.2 * np.ones_like(t), np.full_like(t, y_base - 0.2), np.full_like(t, 0.3)])
        points[5] = np.column_stack([0.2 * np.ones_like(t), np.full_like(t, y_base - 0.2), np.full_like(t, 0.3)])

    elif action == "falling":
        y_base = 1.8
        # Caduta rapida accelerata verso il basso
        fall_t = np.clip(t - 0.5, 0, None)
        z_fall = np.maximum(1.2 - 0.5 * 9.8 * fall_t**2, 0.2)
        points[0] = np.column_stack([np.zeros_like(t), y_base + 0.3*(1.2 - z_fall), z_fall])
        points[1] = np.column_stack([np.zeros_like(t), y_base + 0.5*(1.2 - z_fall), z_fall + 0.3])
        points[2] = np.column_stack([0.4 * np.ones_like(t), np.full_like(t, y_base), z_fall + 0.1])
        points[3] = np.column_stack([-0.4 * np.ones_like(t), np.full_like(t, y_base), z_fall + 0.1])
        points[4] = np.column_stack([-0.2 * np.ones_like(t), np.full_like(t, y_base - 0.2), np.minimum(0.3, z_fall)])
        points[5] = np.column_stack([0.2 * np.ones_like(t), np.full_like(t, y_base - 0.2), np.minimum(0.3, z_fall)])

    return points, rcs

# =====================================================================
# 3. GENERAZIONE SEGNALE IQ SINTETICO ACCONEER A121
# =====================================================================
def synthesize_acconeer_iq(points, rcs, range_bins=NUM_RANGE_BINS, start_range=START_RANGE, res=RANGE_RESOLUTION):
    """
    Modella il segnale IQ acquisito dal radar Acconeer A121.
    S(sweep, range_bin) = Sum_k [ sqrt(RCS_k) * exp(-j * 4 * pi * R_k / lambda) * pulse_envelope ]
    """
    num_scatterers, num_sweeps, _ = points.shape
    range_axis = start_range + np.arange(range_bins) * res
    iq_matrix = np.zeros((num_sweeps, range_bins), dtype=complex)

    for idx_sweep in range(num_sweeps):
        for k in range(num_scatterers):
            pos = points[k, idx_sweep, :]
            # Distanza euclidea dal radar fisso situato in (0,0,1.0)
            R_k = np.sqrt(pos[0]**2 + pos[1]**2 + (pos[2] - 1.0)**2)
            
            # Sfasamento coerente del segnale IQ
            phase = -4.0 * np.pi * R_k / WAVELENGTH
            complex_amplitude = np.sqrt(rcs[k]) * np.exp(1j * phase)
            
            # Inviluppo dell'impulso radar gaussian-shaped lungo i range bin
            pulse_envelope = np.exp(-0.5 * ((range_axis - R_k) / (res * 1.2)) ** 2)
            
            iq_matrix[idx_sweep, :] += complex_amplitude * pulse_envelope

    # Aggiunta rumore termico/fase AWGN
    noise = (np.random.normal(0, NOISE_STD, iq_matrix.shape) + 
             1j * np.random.normal(0, NOISE_STD, iq_matrix.shape))
    return iq_matrix + noise

# =====================================================================
# 4. TRASFORMATA RANGE-DOPPLER (RDM)
# =====================================================================
def compute_range_doppler_map(iq_frame):
    """
    Calcola la Range-Doppler Map applicando la 1D-FFT lungo lo slow-time (sweep).
    """
    # 1. Rimuove il clutter statico (sottrazione della media lungo gli sweep)
    iq_zero_mean = iq_frame - np.mean(iq_frame, axis=0, keepdims=True)
    
    # 2. Windowing di Hann lungo lo slow-time per abbattere i lobi laterali
    window = np.hanning(iq_frame.shape[0])[:, np.newaxis]
    iq_windowed = iq_zero_mean * window
    
    # 3. FFT 1D lungo la dimensione degli sweep
    rdm = np.fft.fftshift(np.fft.fft(iq_windowed, axis=0), axes=0)
    
    # Power Spectral Density (Spettro di Potenza) in dB
    rdm_power = np.abs(rdm)**2
    return rdm_power

# =====================================================================
# 5. ESTRAZIONE DELLE FEATURE (HAND-CRAFTED FEATURES)
# =====================================================================
def extract_handcrafted_features(rdm_stack):
    """
    Estrae feature rilevanti da una sequenza temporale di Mappe Range-Doppler.
    """
    features = []
    
    for rdm in rdm_stack:
        total_energy = np.sum(rdm)
        if total_energy == 0:
            total_energy = 1e-12
        
        # Proiezione lungo l'asse Doppler e l'asse Range
        doppler_profile = np.sum(rdm, axis=1) # Asse 0: Doppler
        range_profile = np.sum(rdm, axis=0)   # Asse 1: Range
        
        # Normalizzazione probabilità
        p_doppler = doppler_profile / np.sum(doppler_profile)
        p_range = range_profile / np.sum(range_profile)
        
        # Feature Doppler
        num_doppler_bins = len(doppler_profile)
        doppler_bins = np.arange(-num_doppler_bins//2, num_doppler_bins//2)
        
        doppler_centroid = np.sum(doppler_bins * p_doppler)
        doppler_spread = np.sqrt(np.sum(((doppler_bins - doppler_centroid)**2) * p_doppler))
        max_doppler_bin = doppler_bins[np.argmax(doppler_profile)]
        
        # Entropia spettrale (misura la turbolenza/complessità del movimento)
        spectral_entropy = -np.sum(p_doppler * np.log2(p_doppler + 1e-12))
        
        # High Doppler Energy Ratio (Fondamentale per identificare le CADUTE)
        high_doppler_mask = np.abs(doppler_bins) > (num_doppler_bins * 0.25)
        high_doppler_ratio = np.sum(doppler_profile[high_doppler_mask]) / total_energy
        
        # Feature Spaziali (Range)
        range_bins_idx = np.arange(len(range_profile))
        mean_range = np.sum(range_bins_idx * p_range)
        
        features.append([
            total_energy,
            doppler_centroid,
            doppler_spread,
            max_doppler_bin,
            spectral_entropy,
            high_doppler_ratio,
            mean_range
        ])
        
    # Aggregazione temporale delle feature (Media e Deviazione Standard sul campione)
    feat_arr = np.array(features)
    mean_feats = np.mean(feat_arr, axis=0)
    std_feats = np.std(feat_arr, axis=0)
    max_feats = np.max(feat_arr, axis=0)
    
    return np.hstack([mean_feats, std_feats, max_feats])

# =====================================================================
# 6. ESECUZIONE PIPELINE COMPLETA E TRAINING RANDOM FOREST
# =====================================================================
def run_pipeline(samples_per_action=100):
    print("--- 1. Generazione Dataset Sintetico Acconeer A121 ---")
    X = []
    y = []
    
    for label_idx, action in enumerate(ACTIONS):
        print(f"Generazione campioni per la classe: {action}...")
        for _ in range(samples_per_action):
            # Aggiunge una leggera variazione casuale alla durata e alla velocità del movimento
            dur = np.random.uniform(1.8, 2.2)
            points, rcs = generate_mocap_trajectory(action, duration_sec=dur)
            
            # Generazione del segnale IQ
            iq_data = synthesize_acconeer_iq(points, rcs)
            
            # Segmentazione in frame consecutivi per le Range-Doppler Maps
            rdm_stack = []
            num_frames = iq_data.shape[0] // NUM_SWEEPS_PER_FRAME
            for f in range(num_frames):
                frame_iq = iq_data[f*NUM_SWEEPS_PER_FRAME : (f+1)*NUM_SWEEPS_PER_FRAME, :]
                rdm = compute_range_doppler_map(frame_iq)
                rdm_stack.append(rdm)
                
            # Estrazione feature
            sample_features = extract_handcrafted_features(rdm_stack)
            X.append(sample_features)
            y.append(label_idx)
            
    X = np.array(X)
    print(X)
    y = np.array(y)
    
    print(f"\nDataset Generato. Shape Matrice Features: {X.shape}, Shape Etichette: {y.shape}")
    
    # Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print("\n--- 2. Addestramento Classificatore Random Forest ---")
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42,min_samples_leaf=30)
    clf.fit(X_train, y_train)
    
    # Valutazione Modello
    y_pred = clf.predict(X_test)
    
    print("\n--- 3. Risultati della Classificazione in training ---")
    print(classification_report(y_test, y_pred, target_names=ACTIONS))

    # Confusion matrix
    cfm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix on the test dataset
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cfm, display_labels=ACTIONS)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)

    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_pipeline(samples_per_action=300)