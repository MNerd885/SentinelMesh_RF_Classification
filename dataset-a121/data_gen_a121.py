import numpy as np
import pandas as pd

# =====================================================================
# 1. Physical and electromagnetic paremeters for the Acconeer A121
# =====================================================================
FC = 60e9                 # Carrier frequency: 60 GHz
C = 3.0e8                 # Light speed (m/s)
WAVELENGTH = C / FC       # Wavelenght (5 mm)
SWEEP_RATE = 400          # Sweep frequency (Hz) - slow-time
NUM_SWEEPS_PER_FRAME = 64 # Number of sweeps for a single Range-Doppler map
RANGE_RESOLUTION = 0.025  # Resolution/distance step (2.5 cm)
NUM_RANGE_BINS = 40       # Number of distance bins
START_RANGE = 0.5         # Minimal measured distance (m)
NOISE_STD = 0.05          # Std deviation of thermal noise/AWGN phase

ACTIONS = ["walking", "bending", "drinking", "lying", "sitting", "falling"]

# =====================================================================
# 2. SIMULATORE CINEMATICO (STILE CMU MOCAP DATASET)
# =====================================================================
def generate_mocap_trajectory(action: str, duration_sec : float, fs :float):
    """
    Modello cinematico a 7 scatterer:
    0: Pelvis (Root)
    1: Torso
    2: Head
    3: Right Hand
    4: Left Hand
    5: Left foot
    6: Right foot
    """
    
    num_samples = int(duration_sec * fs)
    t = np.linspace(0, duration_sec, num_samples)

    # A 3D matrix 7x250x3 containing for every point i its evolution for a time of num_samples
    # given by the evolution of three coordinates (x,y,z)
    points = np.zeros((7, num_samples, 3))

    # Proportional RCS to the body segments 
    rcs = np.array([1.0, 0.8, 0.3, 0.15, 0.15, 0.2, 0.2])
    
    if action == "walking":
        y_base = 2.5 - 0.6 * t  # Linear law by which the stickman moves forward
        # Tronco
        points[0] = np.column_stack([np.zeros_like(t), y_base, np.full_like(t, 1.0)])  # Bacino
        points[1] = np.column_stack([np.zeros_like(t), y_base, np.full_like(t, 1.4)])  # Torso
        points[2] = np.column_stack([np.zeros_like(t), y_base, np.full_like(t, 1.7)])  # Testa
        # Mani (oscillano in controfase)
        points[3] = np.column_stack([ 0.3*np.ones_like(t), y_base + 0.3*np.cos(2*np.pi*1.5*t), 0.9 - 0.1*np.sin(2*np.pi*1.5*t)])
        points[4] = np.column_stack([-0.3*np.ones_like(t), y_base - 0.3*np.cos(2*np.pi*1.5*t), 0.9 + 0.1*np.sin(2*np.pi*1.5*t)])
        # Piedi (passi alternati, non scendono sotto Z=0)
        points[5] = np.column_stack([-0.2*np.ones_like(t), y_base + 0.3*np.sin(2*np.pi*1.5*t), np.maximum(0, 0.2*np.cos(2*np.pi*1.5*t))])
        points[6] = np.column_stack([ 0.2*np.ones_like(t), y_base - 0.3*np.sin(2*np.pi*1.5*t), np.maximum(0, -0.2*np.cos(2*np.pi*1.5*t))])

    elif action == "sitting":
        # It sits smoothly on the first 70% of time
        prog = np.clip(t / (duration_sec * 0.7), 0, 1)
        # Smoothening of the animation (ease-in-out)
        prog = 0.5 * (1 - np.cos(np.pi * prog))
        
        y_base = 2.0
        # The pelvis goes down and it goes a bit backwards
        points[0] = np.column_stack([np.zeros_like(t), y_base + 0.2*prog, 1.0 - 0.5*prog])
        points[1] = np.column_stack([np.zeros_like(t), y_base + 0.2*prog, 1.4 - 0.5*prog])
        points[2] = np.column_stack([np.zeros_like(t), y_base + 0.2*prog, 1.7 - 0.5*prog])
        # The hands lay on the knees
        points[3] = np.column_stack([ 0.3*np.ones_like(t), y_base - 0.2*prog, 0.9 - 0.3*prog])
        points[4] = np.column_stack([-0.3*np.ones_like(t), y_base - 0.2*prog, 0.9 - 0.3*prog])
        # Feet are always still on the ground(Z=0, Y=avanti al bacino)
        points[5] = np.column_stack([-0.2*np.ones_like(t), np.full_like(t, y_base - 0.3), np.zeros_like(t)])
        points[6] = np.column_stack([ 0.2*np.ones_like(t), np.full_like(t, y_base - 0.3), np.zeros_like(t)])

    elif action == "bending":
        # It bends down and then it gets up
        bend = np.sin(np.pi * t / duration_sec)
        y_base = 2.0
        
        # The pelvis goes slightly backwards in order to balance, Z remains almost the same
        points[0] = np.column_stack([np.zeros_like(t), y_base + 0.2*bend, 1.0 - 0.1*bend])
        # The Torso and Head go down and move forward to the radar
        points[1] = np.column_stack([np.zeros_like(t), y_base - 0.4*bend, 1.4 - 0.6*bend])
        points[2] = np.column_stack([np.zeros_like(t), y_base - 0.7*bend, 1.7 - 0.9*bend])
        # Le mani vanno verso terra
        points[3] = np.column_stack([ 0.2*np.ones_like(t), y_base - 0.6*bend, 0.9 - 0.7*bend])
        points[4] = np.column_stack([-0.2*np.ones_like(t), y_base - 0.6*bend, 0.9 - 0.7*bend])
        # Piedi fermi a terra
        points[5] = np.column_stack([-0.2*np.ones_like(t), np.full_like(t, y_base), np.zeros_like(t)])
        points[6] = np.column_stack([ 0.2*np.ones_like(t), np.full_like(t, y_base), np.zeros_like(t)])

    elif action == "falling":
        # Caduta in avanti accelerata dalla gravità
        fall_t = np.clip((t - 0.3) * 1.5, 0, 1)
        fall_prog = fall_t ** 2  # Accelerazione
        y_base = 2.0
        
        points[0] = np.column_stack([np.zeros_like(t), y_base - 0.8*fall_prog, 1.0 - 0.8*fall_prog])
        points[1] = np.column_stack([np.zeros_like(t), y_base - 1.4*fall_prog, 1.4 - 1.2*fall_prog])
        points[2] = np.column_stack([np.zeros_like(t), y_base - 1.7*fall_prog, 1.7 - 1.5*fall_prog])
        points[3] = np.column_stack([ 0.4*np.ones_like(t), y_base - 1.4*fall_prog, 0.9 - 0.7*fall_prog])
        points[4] = np.column_stack([-0.4*np.ones_like(t), y_base - 1.4*fall_prog, 0.9 - 0.7*fall_prog])
        # Piedi fermi (fanno da perno alla caduta)
        points[5] = np.column_stack([-0.2*np.ones_like(t), np.full_like(t, y_base), np.zeros_like(t)])
        points[6] = np.column_stack([ 0.2*np.ones_like(t), np.full_like(t, y_base), np.zeros_like(t)])

    elif action == "lying":
        # Soggetto disteso a terra con micro-respirazione sul torso a 0.3Hz
        y_base = 1.0
        breath = 0.02 * np.sin(2 * np.pi * 0.3 * t)
        
        points[5] = np.column_stack([-0.2*np.ones_like(t), np.full_like(t, y_base), np.full_like(t, 0.1)]) # Piedi vicini al radar
        points[6] = np.column_stack([ 0.2*np.ones_like(t), np.full_like(t, y_base), np.full_like(t, 0.1)])
        points[0] = np.column_stack([np.zeros_like(t), np.full_like(t, y_base + 0.9), np.full_like(t, 0.1)]) # Bacino
        points[1] = np.column_stack([np.zeros_like(t), np.full_like(t, y_base + 1.4), 0.1 + breath]) # Torso respira
        points[3] = np.column_stack([ 0.4*np.ones_like(t), np.full_like(t, y_base + 1.3), np.full_like(t, 0.1)])
        points[4] = np.column_stack([-0.4*np.ones_like(t), np.full_like(t, y_base + 1.3), np.full_like(t, 0.1)])
        points[2] = np.column_stack([np.zeros_like(t), np.full_like(t, y_base + 1.7), np.full_like(t, 0.1)]) # Testa
        
    elif action == "drinking":
        # Solo il braccio si muove verso la testa
        y_base = 2.0
        drink = np.sin(np.pi * 1.5 * t / duration_sec)
        
        points[0] = np.column_stack([np.zeros_like(t), np.full_like(t, y_base), np.full_like(t, 1.0)])
        points[1] = np.column_stack([np.zeros_like(t), np.full_like(t, y_base), np.full_like(t, 1.4)])
        points[2] = np.column_stack([np.zeros_like(t), np.full_like(t, y_base), np.full_like(t, 1.7)])
        points[3] = np.column_stack([0.2*(1-drink), y_base - 0.2*drink, 0.9 + 0.7*drink]) # Mano DX va alla bocca
        points[4] = np.column_stack([-0.3*np.ones_like(t), np.full_like(t, y_base), np.full_like(t, 0.9)])
        points[5] = np.column_stack([-0.2*np.ones_like(t), np.full_like(t, y_base), np.zeros_like(t)])
        points[6] = np.column_stack([ 0.2*np.ones_like(t), np.full_like(t, y_base), np.zeros_like(t)])

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
            complex_amplitude = (np.sqrt(rcs[k]) / (R_k**2) ) * np.exp(1j * phase)
            
            # Inviluppo dell'impulso radar gaussian-shaped lungo i range bin
            pulse_envelope = np.exp(-0.5 * ((range_axis - R_k) / (res * 1.2)) ** 2)
            
            iq_matrix[idx_sweep, :] += complex_amplitude * pulse_envelope

    # Aggiunta rumore termico/fase AWGN
    noise = (np.random.normal(0, NOISE_STD, iq_matrix.shape) + 
             1j * np.random.normal(0, NOISE_STD, iq_matrix.shape))
    return iq_matrix + noise

# =====================================================================
# 3.5. PROFILO DI RANGE GREZZO (NESSUN CLUTTER REMOVAL)
# =====================================================================
def compute_raw_range_profile(iq_data_full):
    """
    Profilo di riflettività per range bin calcolato SENZA sottrazione
    della media lungo lo slow-time. A differenza della Range-Doppler Map
    (pensata per isolare il moto), questo profilo preserva l'estensione
    spaziale del corpo, inclusi gli scatterer perfettamente fermi che la
    clutter removal cancellerebbe.
 
    iq_data_full: matrice IQ dell'intero trial (tutti gli sweep), non un
    singolo frame — la posizione del corpo non cambia frame-per-frame
    come il Doppler, quindi mediare su più sweep dà anche una stima più
    pulita rispetto al rumore.
    """
    return np.mean(np.abs(iq_data_full) ** 2, axis=0)

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
def extract_handcrafted_features(rdm_stack, iq_data_full):
    """
    Estrae feature rilevanti da:
    - una sequenza temporale di Mappe Range-Doppler (rdm_stack) -> feature
      di MOTO (Doppler), calcolate per frame e aggregate su mean/std/max;
    - il segnale IQ grezzo dell'intero trial (iq_data_full) -> feature di
      POSIZIONE/POSTURA, calcolate una sola volta per trial, perché la
      RDM ha già cancellato l'informazione spaziale statica.
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
        
        # Feature Spaziali (Range) - calcolate dalla RDM (post clutter-removal):
        # utili come proxy di "dove si trova ciò che si muove", non come
        # estensione spaziale del corpo (per quella vedi raw_* più sotto).
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
    # per far in modo che la RF abbia un dataset a lunghezza fissa:
    # - mean_feats tells me how a feature in 'features' varies in a trial on average.
    # - std_feat tells me how much that feature changes over time. A still posture gives constant values, 
    #   walking and falling don't.
    # - max_feats meybe to substitute with 90th percentile.

    feat_arr = np.array(features)
    # axis=0 "collapses" the rows col by col, e.g. if I have (30,7) it becomes -> (7,)
    mean_feats = np.mean(feat_arr, axis=0)
    std_feats = np.std(feat_arr, axis=0)
    max_feats = np.max(feat_arr, axis=0)
    
    # --- Feature di POSTURA, dal profilo di range grezzo (una volta per trial) ---
    raw_profile = compute_raw_range_profile(iq_data_full)
    
    # Normalizzazione probabilità
    p_raw = raw_profile / np.sum(raw_profile)
    
    raw_idx = np.arange(len(raw_profile))
 
    raw_mean_range = np.sum(raw_idx * p_raw)
    raw_range_spread = np.sqrt(np.sum(((raw_idx - raw_mean_range) ** 2) * p_raw))
    occupied = raw_idx[raw_profile > 0.05 * raw_profile.max()]
    raw_range_span = occupied.max() - occupied.min() if occupied.size else 0
 
    # Escursione del "centro di massa" del moto nel corso del trial
    # (colonna 6 = mean_range per frame): cattura oscillazioni (drinking)
    # e spostamenti monotoni (falling/sitting) allo stesso modo.
    range_excursion = feat_arr[:, 6].max() - feat_arr[:, 6].min()
 
    posture_feats = np.array([raw_mean_range, raw_range_spread, raw_range_span, range_excursion])
 
    return np.hstack([mean_feats, std_feats, max_feats, posture_feats])

# =====================================================================
# 6. ESECUZIONE PIPELINE COMPLETA E CREAZIONE DATASET
# =====================================================================
def inspect_features_by_class(X, y):
    """
    Diagnostica rapida: confronta mean/std/max delle feature per classe.
    Indici delle feature per frame (7 totali): 
    0=total_energy, 1=doppler_centroid, 2=doppler_spread, 3=max_doppler_bin,
    4=spectral_entropy, 5=high_doppler_ratio, 6=mean_range (post clutter-removal)
    Nel vettore finale: mean=0-6, std=7-13, max=14-20,
    poi 21=raw_mean_range, 22=raw_range_spread, 23=raw_range_span, 24=range_excursion
    """
    for label_idx, action in enumerate(ACTIONS):
        mask = (y == label_idx)
        print(f"{action:10s}  "
              f"raw_mean_range={X[mask,21].mean():7.3f}  "
              f"raw_range_spread={X[mask,22].mean():6.3f}  "
              f"raw_range_span={X[mask,23].mean():6.3f}  "
              f"range_excursion={X[mask,24].mean():6.3f}")

def run_pipeline(samples_per_action):
    print("--- 1. Generazione Dataset Sintetico Acconeer A121 ---")
    X = []
    Y = []
    
    for label_idx, action in enumerate(ACTIONS):
        print(f"Generazione campioni per la classe: {action}...")
        for _ in range(samples_per_action):

            # Aggiunge una leggera variazione casuale alla durata e alla velocità del movimento
            dur = np.random.uniform(1.8, 3)
            points, rcs = generate_mocap_trajectory(action, duration_sec=dur, fs=SWEEP_RATE)
            
            # Generazione del segnale IQ
            iq_data = synthesize_acconeer_iq(points, rcs)
            
            # Segmentazione in frame consecutivi per le Range-Doppler Maps
            rdm_stack = []
            num_frames = iq_data.shape[0] // NUM_SWEEPS_PER_FRAME
            for f in range(num_frames):
                frame_iq = iq_data[f*NUM_SWEEPS_PER_FRAME : (f+1)*NUM_SWEEPS_PER_FRAME, :]
                rdm = compute_range_doppler_map(frame_iq)
                rdm_stack.append(rdm)
                
            # Features and target labels
            sample_features = extract_handcrafted_features(rdm_stack,iq_data)
            X.append(sample_features)
            Y.append(label_idx)
            
    X = np.array(X)
    Y = np.array(Y)

    # Diagnostica movimenti
    inspect_features_by_class(X,Y)
    
    print(f"\nDataset Generato. Shape Matrice Features: {X.shape}, Shape Etichette: {Y.shape}")

    # Conversion in pandas DataFrame and saving on .csv file
    dfx = pd.DataFrame(X, columns=[
        "mean_total_energy",
        "mean_doppler_centroid",
        "mean_doppler_spread",
        "mean_max_doppler_bin",
        "mean_spectral_entropy",
        "mean_high_doppler_ratio",
        "mean_mean_range",
        "std_total_energy",
        "std_doppler_centroid",
        "std_doppler_spread",
        "std_max_doppler_bin",
        "std_spectral_entropy",
        "std_high_doppler_ratio",
        "std_mean_range",
        "mean_total_energy",
        "max_doppler_centroid",
        "max_doppler_spread",
        "max_max_doppler_bin",
        "max_spectral_entropy",
        "max_high_doppler_ratio",
        "max_mean_range",
        "raw_mean_range", 
        "raw_range_spread", 
        "raw_range_span", 
        "range_excursion"
    ])
    dfx.to_csv('dataset-a121/synth_dataset_A121.csv', index=False)
    dfy = pd.DataFrame(Y, columns=["sequence of motions"])
    dfy.to_csv("dataset-a121/synth_dataset_A121_targets.csv", index=False)

if __name__ == "__main__":
    run_pipeline(samples_per_action=200)