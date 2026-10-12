import numpy as np
import pandas as pd

# =====================================================================
# 1. Physical and electromagnetic paremeters for the Acconeer A121
# =====================================================================
FC = 60e9                 # Carrier frequency: 60 GHz
C = 3.0e8                 # Light speed (m/s)
LAMBDA = C / FC           # Wavelenght (5 mm)
FRAME_RATE = 10.0         # Hz - how often the radar wakes up and measures a frame (HW limit: 5Hz min - 10 Hz max)
SWEEP_RATE = 1000         # Hz - slow-time rate of the sweeps INSIDE one frame (burst). Sets max
                          # unambiguous speed: v_max = SWEEP_RATE * LAMBDA / 4  (=1.25 m/s @ 1 kHz)
NUM_SWEEPS_PER_FRAME = 8  # Sweeps per frame (= Doppler bins of the Range-Doppler map)


# --- Range configuration + sensor buffer budget ---
NUM_RANGE_BINS = 16     # Number of distance bins
START_RANGE = 0.8         # Minimal measured distance (m)
RANGE_STEP = 0.14         # Distance between points (m) -> window 0.8 .. 2.9 m, covers the whole scene
PULSE_SIGMA = 1.2 * RANGE_STEP  # Pulse width, matched to the step (as the profile would be chosen on the A121)
BYTES_PER_POINT = 4       # Complex int16 (I + Q): acc_int16_complex_t in the Acconeer SDK (not in the datasheet)
BUFFER_BYTES = 512        # Self-imposed budget per frame (the A121 sensor buffer holds up to 4095 points)
NOISE_STD = 0.05          # Std deviation of thermal noise/AWGN phase

ACTIONS = ["walking", "bending", "drinking", "lying", "sitting", "falling"]

# =====================================================================
# 2. KINEMATIC SIMULATOR (CMU MOCAP DATASET STYLE)
# =====================================================================
def generate_mocap_trajectory(action: str, duration_sec : float, fs :float):
    """
    Kinematic model with 7 scatterers:
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

    # A 3D matrix 7xnum_samplesx3 containing for every point i its evolution for a time of num_samples
    # given by the evolution of three coordinates (x,y,z)
    points = np.zeros((7, num_samples, 3))

    # Proportional RCS to the body segments 
    rcs = np.array([1.0, 0.8, 0.3, 0.15, 0.15, 0.2, 0.2])
    
    if action == "walking":
        y_base = 2.5 - 0.6 * t  # Linear law by which the stickman moves forward
        # Trunk
        points[0] = np.column_stack([np.zeros_like(t), y_base, np.full_like(t, 1.0)])  # Pelvis
        points[1] = np.column_stack([np.zeros_like(t), y_base, np.full_like(t, 1.4)])  # Torso
        points[2] = np.column_stack([np.zeros_like(t), y_base, np.full_like(t, 1.7)])  # Head
        # Hands (swing in antiphase)
        points[3] = np.column_stack([ 0.3*np.ones_like(t), y_base + 0.3*np.cos(2*np.pi*1.5*t), 0.9 - 0.1*np.sin(2*np.pi*1.5*t)])
        points[4] = np.column_stack([-0.3*np.ones_like(t), y_base - 0.3*np.cos(2*np.pi*1.5*t), 0.9 + 0.1*np.sin(2*np.pi*1.5*t)])
        # Feet (alternating steps, never go below Z=0)
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
        # Feet are always still on the ground(Z=0, Y=in front of the pelvis)
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
        # The hands go towards the ground
        points[3] = np.column_stack([ 0.2*np.ones_like(t), y_base - 0.6*bend, 0.9 - 0.7*bend])
        points[4] = np.column_stack([-0.2*np.ones_like(t), y_base - 0.6*bend, 0.9 - 0.7*bend])
        # Feet still on the ground
        points[5] = np.column_stack([-0.2*np.ones_like(t), np.full_like(t, y_base), np.zeros_like(t)])
        points[6] = np.column_stack([ 0.2*np.ones_like(t), np.full_like(t, y_base), np.zeros_like(t)])

    elif action == "falling":
        # Forward fall accelerated by gravity
        fall_t = np.clip((t - 0.3) * 1.5, 0, 1)
        fall_prog = fall_t ** 2  # Acceleration
        y_base = 2.0
        
        points[0] = np.column_stack([np.zeros_like(t), y_base - 0.8*fall_prog, 1.0 - 0.8*fall_prog])
        points[1] = np.column_stack([np.zeros_like(t), y_base - 1.4*fall_prog, 1.4 - 1.2*fall_prog])
        points[2] = np.column_stack([np.zeros_like(t), y_base - 1.7*fall_prog, 1.7 - 1.5*fall_prog])
        points[3] = np.column_stack([ 0.4*np.ones_like(t), y_base - 1.4*fall_prog, 0.9 - 0.7*fall_prog])
        points[4] = np.column_stack([-0.4*np.ones_like(t), y_base - 1.4*fall_prog, 0.9 - 0.7*fall_prog])
        # Feet still (they act as the pivot of the fall)
        points[5] = np.column_stack([-0.2*np.ones_like(t), np.full_like(t, y_base), np.zeros_like(t)])
        points[6] = np.column_stack([ 0.2*np.ones_like(t), np.full_like(t, y_base), np.zeros_like(t)])

    elif action == "lying":
        # Subject lying on the ground with micro-breathing on the torso at 0.3Hz
        y_base = 1.0
        breath = 0.02 * np.sin(2 * np.pi * 0.3 * t)
        
        points[5] = np.column_stack([-0.2*np.ones_like(t), np.full_like(t, y_base), np.full_like(t, 0.1)]) # Feet close to the radar
        points[6] = np.column_stack([ 0.2*np.ones_like(t), np.full_like(t, y_base), np.full_like(t, 0.1)])
        points[0] = np.column_stack([np.zeros_like(t), np.full_like(t, y_base + 0.9), np.full_like(t, 0.1)]) # Pelvis
        points[1] = np.column_stack([np.zeros_like(t), np.full_like(t, y_base + 1.4), 0.1 + breath]) # Torso breathes
        points[3] = np.column_stack([ 0.4*np.ones_like(t), np.full_like(t, y_base + 1.3), np.full_like(t, 0.1)])
        points[4] = np.column_stack([-0.4*np.ones_like(t), np.full_like(t, y_base + 1.3), np.full_like(t, 0.1)])
        points[2] = np.column_stack([np.zeros_like(t), np.full_like(t, y_base + 1.7), np.full_like(t, 0.1)]) # Head
        
    elif action == "drinking":
        # Only the arm moves towards the head
        y_base = 2.0
        drink = np.sin(np.pi * 1.5 * t / duration_sec)
        
        points[0] = np.column_stack([np.zeros_like(t), np.full_like(t, y_base), np.full_like(t, 1.0)])
        points[1] = np.column_stack([np.zeros_like(t), np.full_like(t, y_base), np.full_like(t, 1.4)])
        points[2] = np.column_stack([np.zeros_like(t), np.full_like(t, y_base), np.full_like(t, 1.7)])
        points[3] = np.column_stack([0.2*(1-drink), y_base - 0.2*drink, 0.9 + 0.7*drink]) # Right hand goes to the mouth
        points[4] = np.column_stack([-0.3*np.ones_like(t), np.full_like(t, y_base), np.full_like(t, 0.9)])
        points[5] = np.column_stack([-0.2*np.ones_like(t), np.full_like(t, y_base), np.zeros_like(t)])
        points[6] = np.column_stack([ 0.2*np.ones_like(t), np.full_like(t, y_base), np.zeros_like(t)])

    return points, rcs

# =====================================================================
# 3. ACCONEER A121 SYNTHETIC IQ SIGNAL GENERATION
# =====================================================================
def check_hardware_budget():
    """Fails if the configuration does not fit the hardware constraints, prints a summary."""
    n_points = NUM_SWEEPS_PER_FRAME * NUM_RANGE_BINS
    n_bytes = n_points * BYTES_PER_POINT
    assert n_bytes <= BUFFER_BYTES, f"Frame = {n_bytes} B > buffer budget {BUFFER_BYTES} B"
    assert SWEEP_RATE % FRAME_RATE == 0, "SWEEP_RATE must be a multiple of FRAME_RATE"
    assert NUM_SWEEPS_PER_FRAME / SWEEP_RATE <= 1.0 / FRAME_RATE, "Burst longer than the frame period"
 
    v_max = SWEEP_RATE * LAMBDA / 4.0
    v_res = (SWEEP_RATE / NUM_SWEEPS_PER_FRAME) * LAMBDA / 2.0
    stop_range = START_RANGE + (NUM_RANGE_BINS - 1) * RANGE_STEP
    print(f"Frame: {NUM_SWEEPS_PER_FRAME} sweeps x {NUM_RANGE_BINS} points = {n_points} points = {n_bytes} B (budget {BUFFER_BYTES} B)")
    print(f"Frame rate {FRAME_RATE:g} Hz | burst {1e3*NUM_SWEEPS_PER_FRAME/SWEEP_RATE:.1f} ms @ {SWEEP_RATE} Hz sweep rate")
    print(f"Doppler: v_max = +/-{v_max:.2f} m/s, resolution = {v_res:.3f} m/s per bin")
    print(f"Range window: {START_RANGE:.2f} .. {stop_range:.2f} m, step {100*RANGE_STEP:.0f} cm")


def synthesize_acconeer_iq(points, rcs):
    """
    Models the IQ signal acquired by the Acconeer A121 as a sequence of FRAMES.
    The radar wakes up every 1/FRAME_RATE s and measures a burst of NUM_SWEEPS_PER_FRAME
    consecutive sweeps spaced by 1/SWEEP_RATE s; between frames nothing is measured.
    S(sweep, range_bin) = Sum_k [ sqrt(RCS_k)/R_k^2 * exp(-j * 4 * pi * R_k / lambda) * pulse_envelope ]
 
    points: (7, num_samples, 3) trajectory sampled at SWEEP_RATE.
    Returns the IQ array with shape (num_frames, NUM_SWEEPS_PER_FRAME, NUM_RANGE_BINS).
    """
    num_samples = points.shape[1]
    # how many trajectory samples are between two frames start point
    num_s_2frames = int(round(SWEEP_RATE / FRAME_RATE))
    num_frames = (num_samples - NUM_SWEEPS_PER_FRAME) // num_s_2frames + 1
    
    # Index of the trajectory sample used by sweep s of frame f: (f, s), so we obtain a matrix of framexsweep
    # in this way I extract only the samples acquired when the radar is up.
    # It is done by summing the column np.arange(num_frames)[:, None] * num_s_2frames to
    # the row np.arange(NUM_SWEEPS_PER_FRAME)[None, :]
    idx = np.arange(num_frames)[:, None] * num_s_2frames + np.arange(NUM_SWEEPS_PER_FRAME)[None, :]
    
    # By putting idx in the points of the movements I obtain the following structure
    # (Scatterer, f, s, (x,y,z))
    pos = points[:, idx, :]                                    

    # Euclidean distance from the radar positioned at (0,0,1.0)
    R = np.sqrt(pos[...,0]**2 + pos[...,1]**2 + (pos[...,2] - 1.0)**2) # (Scatterer, f, s)
    
    # Coherent IQ amplitude/phase of each scatterer
    phase = -4.0 * np.pi * R / LAMBDA
    complex_amplitude = (np.sqrt(rcs)[:, None, None] / R**2) * np.exp(1j * phase)  # (K, F, S)
    
    # Gaussian-shaped radar pulse envelope along the range points
    range_axis = START_RANGE + np.arange(NUM_RANGE_BINS) * RANGE_STEP       # range bins(B,)
    # range_axis - R[..., None] calculates the distance of every scatterer from every bin, the result
    # has the form # (K, F, S, B) --> (Scatterer, f, s, B)
    pulse_envelope = np.exp(-0.5 * ((range_axis - R[..., None]) / PULSE_SIGMA) ** 2)  # (K, F, S, B)
    # Here I sum the 7 contribuitions as complex values. If two points fall on the same bin
    # they can interfere, as in a real radar.
    iq_matrix = np.sum(complex_amplitude[..., None] * pulse_envelope, axis=0)   # (f, s, B)

    # QUICK Note: In NumPy, the expression arr[..., None] (or arr[:, None]) uses None as 
    # an alias for np.newaxis to add a new axis of length 1 to the array. 
    # This operation is primarily used to enable broadcasting between arrays of 
    # different dimensions.  For example, if you have a 1D array and want to perform 
    # an operation with a 2D array, adding None reshapes the 1D array to a column 
    # vector (or row vector) so that NumPy can automatically expand it to match the 
    # other array's shape without explicit loops or reshape calls.


    # Adding thermal noise / AWGN additive noise
    noise = (np.random.normal(0, NOISE_STD, iq_matrix.shape) + 
             1j * np.random.normal(0, NOISE_STD, iq_matrix.shape))
    return iq_matrix + noise

def range_window_coverage(points, rcs):
    """Fraction of the received power (~RCS/R^4) whose scatterers fall inside the range window."""
    R = np.sqrt(points[..., 0]**2 + points[..., 1]**2 + (points[..., 2] - 1.0)**2)

    stop_range = START_RANGE + (NUM_RANGE_BINS - 1) * RANGE_STEP

    inside = (R >= START_RANGE - PULSE_SIGMA) & (R <= stop_range + PULSE_SIGMA)

    w = rcs[:, None] / R**4

    return float((w * inside).sum() / w.sum())

# =====================================================================
# 3.5. RAW RANGE PROFILE (NO CLUTTER REMOVAL)
# =====================================================================
def compute_raw_range_profile(iq_data_full):
    """
    Reflectivity profile per range bin computed WITHOUT subtracting the
    mean along the slow-time. Unlike the Range-Doppler Map (designed to
    isolate motion), this profile preserves the spatial extent of the
    body, including perfectly still scatterers that clutter removal
    would cancel.
 
    iq_data_full: IQ matrix of the entire trial (all sweeps), not a
    single frame — the body position does not change frame-by-frame
    like the Doppler does, so averaging over more sweeps also gives a
    cleaner estimate with respect to noise.
    """
    return np.mean(np.abs(iq_data_full) ** 2, axis=0)

# =====================================================================
# 4. RANGE-DOPPLER TRANSFORM (RDM)
# =====================================================================
def compute_range_doppler_map(iq_frame):
    """
    Computes the Range-Doppler Map by applying a 1D FFT along the slow-time (sweeps).
    """
    # 1. Removes static clutter (all stationary or slow-moving joints) (subtraction of the mean 
    # along the sweeps/Range)
    iq_zero_mean = iq_frame - np.mean(iq_frame, axis=0, keepdims=True)
    
    # 2. Hann's windowing along the slow-time to cut lateral lobes
    window = np.hanning(iq_frame.shape[0])[:, np.newaxis]
    iq_windowed = iq_zero_mean * window
    
    # 3. 1D FFT along the sweep dimension and in this way with np.fft.fftshift I can shift the 
    # zero-frequency component index to the center of the spectrum, useful to know if the person
    # is going near the radar or not.
    rdm = np.fft.fftshift(np.fft.fft(iq_windowed, axis=0), axes=0)
    
    # Power Spectral Density [dB] is needed to analyze the energy at each distance given a velocity.
    rdm_power = np.abs(rdm)**2
    return rdm_power


# =====================================================================
# 5. FEATURE EXTRACTION (HAND-CRAFTED FEATURES)
# =====================================================================
def extract_handcrafted_features(rdm_stack, iq_data_full):
    """
    Extracts relevant features from:
    - a temporal sequence of Range-Doppler Maps (rdm_stack) -> MOTION
      (Doppler) features, computed per frame and aggregated as mean/std/max;
    - the raw IQ signal of the entire trial (iq_data_full) -> POSITION/POSTURE
      features, computed only once per trial, because the RDM has already
      cancelled the static spatial information.
    """
    features = []
    
    for rdm in rdm_stack:
        total_energy = np.sum(rdm)
        if total_energy == 0:
            total_energy = 1e-12
        
        # Projection over the Doppler and Range axes: "how much energy there's at each velocity/distance
        # by summing all the distances/velocitites"
        doppler_profile = np.sum(rdm, axis=1) # 0 axis: Doppler
        range_profile = np.sum(rdm, axis=0)   # 1 axis: Range
        
        # Profiles normalization
        p_doppler = doppler_profile / np.sum(doppler_profile)
        p_range = range_profile / np.sum(range_profile)
        
        # Doppler features
        num_doppler_bins = len(doppler_profile)
        # In this way I subdivide the bins such that I have the bin corresponding to
        # 0 velocity at the center (index = 16), the negative bins indicate that I'm 
        # moving farther from the radar, the others indicate that I'm moving 
        # to the radar.
        doppler_bins = np.arange(-num_doppler_bins//2, num_doppler_bins//2)

        # It represents the centroid for each doppler bin which is weighted by how much energy there's in that
        # particular bin
        doppler_centroid = np.sum(doppler_bins * p_doppler)
        # This represent how much a bin deviates from the found centroid fro that bin
        doppler_spread = np.sqrt(np.sum(((doppler_bins - doppler_centroid)**2) * p_doppler))
        # Represents the greatest velocity in that frame
        max_doppler_bin = doppler_bins[np.argmax(doppler_profile)]
        
        # Spectral entropy: measures how much the movement is chaotic/ordered. On a scale of log_2 (NUM_SWEEPS) = log_2 (32) = 5 bits,
        # in [0,5], when it is 0 we have the velocity concentrated on one velocity (simple movement, regular movement);when it is 5 the 
        # energy is distirbuted among all the velocities in an almost uniform way (there's also noise).
        spectral_entropy = -np.sum(p_doppler * np.log2(p_doppler + 1e-12))
        
        # High Doppler Energy Ratio (fundamental to identify FALLS) indicates which movement fraction is really fast.
        high_doppler_mask = np.abs(doppler_bins) > (num_doppler_bins * 0.25)
        high_doppler_ratio = np.sum(doppler_profile[high_doppler_mask]) / total_energy
        
        # Spatial features (Range) - computed from the RDM (post-clutter removal):
        # useful as a proxy (a measured quantity which is not the one I want to measure, but it's correlated)
        # of "where the thing that is moving is", not a spatial extension of the body (for that there's the 
        # raw profile below).
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
        
    # Temporal aggregation of the features (Mean and Standard Deviation over the sample)
    # so that the RF gets a fixed-length dataset:
    # - mean_feats tells me how a feature in 'features' varies in a trial on average.
    # - std_feat tells me how much that feature changes over time. A still posture gives constant values, 
    #   walking and falling don't.
    # - max_feats tells me the most extreme moment. For example the the subject falls (maybe to be substituted with something else).

    feat_arr = np.array(features)
    # axis=0 "collapses" the rows col by col, e.g. if I have (30,7) it becomes -> (7,)
    mean_feats = np.mean(feat_arr, axis=0)
    std_feats = np.std(feat_arr, axis=0)
    max_feats = np.max(feat_arr, axis=0)
    
    # --- POSTURE features, from the raw range profile (once per trial) ---
    raw_profile = compute_raw_range_profile(iq_data_full)
    
    # Probability normalization
    p_raw = raw_profile / np.sum(raw_profile)
    
    raw_idx = np.arange(len(raw_profile))
 
    raw_mean_range = np.sum(raw_idx * p_raw)
    raw_range_spread = np.sqrt(np.sum(((raw_idx - raw_mean_range) ** 2) * p_raw))
    # soglia al 5% del picco: isola solo i bin con riflessione reale
    occupied = raw_idx[raw_profile > 0.05 * raw_profile.max()] 
    # estensione fisica occupata
    raw_range_span = occupied.max() - occupied.min() if occupied.size else 0
 
    # Excursion of the motion's "center of mass" over the trial
    # (column 6 = mean_range per frame): captures oscillations (drinking)
    # and monotonic shifts (falling/sitting) alike.
    range_excursion = feat_arr[:, 6].max() - feat_arr[:, 6].min()
 
    posture_feats = np.array([raw_mean_range, raw_range_spread, raw_range_span, range_excursion])
 
    return np.hstack([mean_feats, std_feats, max_feats, posture_feats])

# =====================================================================
# 6. FULL PIPELINE EXECUTION AND DATASET CREATION
# =====================================================================
def inspect_features_by_class(X, y):
    """
    Quick diagnostics: compares mean/std/max of the features per class.
    Per-frame feature indices (7 in total): 
    0=total_energy, 1=doppler_centroid, 2=doppler_spread, 3=max_doppler_bin,
    4=spectral_entropy, 5=high_doppler_ratio, 6=mean_range (post clutter-removal)
    In the final vector: mean=0-6, std=7-13, max=14-20,
    then 21=raw_mean_range, 22=raw_range_spread, 23=raw_range_span, 24=range_excursion
    """
    for label_idx, action in enumerate(ACTIONS):
        mask = (y == label_idx)
        print(f"{action:10s}  "
              f"raw_mean_range={X[mask,21].mean():7.3f}  "
              f"raw_range_spread={X[mask,22].mean():6.3f}  "
              f"raw_range_span={X[mask,23].mean():6.3f}  "
              f"range_excursion={X[mask,24].mean():6.3f}")

def run_pipeline(samples_per_action):

    check_hardware_budget()
    
    print("--- 1. Generazione Dataset Sintetico Acconeer A121 ---")
    X = []
    Y = []
    
    for label_idx, action in enumerate(ACTIONS):
        print(f"Generazione campioni per la classe: {action}...")
        for n in range(samples_per_action):

            # Adds a slight random variation to the duration and speed of the movement
            dur = np.random.uniform(1.8, 3)
            points, rcs = generate_mocap_trajectory(action, duration_sec=dur, fs=SWEEP_RATE)

            if n == 0:
                print(f"    potenza nella finestra di range: {100*range_window_coverage(points, rcs):.0f}%")
            
            # IQ signal generation
            iq_data = synthesize_acconeer_iq(points, rcs)
            
            # Segmentation into consecutive frames for the Range-Doppler Maps
            rdm_stack = [compute_range_doppler_map(frame_iq) for frame_iq in iq_data]

            # Raw IQ of the trial (all measured sweeps) for the posture features
            iq_data = iq_data.reshape(-1, NUM_RANGE_BINS)
                
            # Features and target labels
            sample_features = extract_handcrafted_features(rdm_stack,iq_data)
            X.append(sample_features)
            Y.append(label_idx)
            
    X = np.array(X)
    Y = np.array(Y)

    # Motion diagnostics
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
        "max_total_energy",
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
    dfx.to_csv('dataset-a121/synth_dataset_A121_2.csv', index=False)
    dfy = pd.DataFrame(Y, columns=["sequence of motions"])
    dfy.to_csv("dataset-a121/synth_dataset_A121_targets_2.csv", index=False)

if __name__ == "__main__":
    run_pipeline(samples_per_action=80)