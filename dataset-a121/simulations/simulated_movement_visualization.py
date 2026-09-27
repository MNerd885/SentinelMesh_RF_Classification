import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

'''
This script has been made to make the simulated movements clear, I tested them and then I
fixed by adding two more joints for the Pelvis and left hand, for a total of 7 joints.
'''

SWEEP_RATE = 100

ACTIONS = ["walking", "bending", "drinking", "lying", "sitting", "falling"]

def generate_mocap_trajectory(action, duration_sec=2.5, fs=SWEEP_RATE):
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
    # RCS proporzionali ai segmenti corporei
    rcs = np.array([1.0, 0.8, 0.3, 0.15, 0.15, 0.2, 0.2])
    
    if action == "walking":
        y_base = 2.5 - 0.6 * t  # Avanzamento
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
        # Ci si siede fluidamente nei primi 70% del tempo
        prog = np.clip(t / (duration_sec * 0.7), 0, 1)
        # Smussamento dell'animazione (ease-in-out)
        prog = 0.5 * (1 - np.cos(np.pi * prog))
        
        y_base = 2.0
        # Il bacino scende e si sposta leggermente indietro
        points[0] = np.column_stack([np.zeros_like(t), y_base + 0.2*prog, 1.0 - 0.5*prog])
        points[1] = np.column_stack([np.zeros_like(t), y_base + 0.2*prog, 1.4 - 0.5*prog])
        points[2] = np.column_stack([np.zeros_like(t), y_base + 0.2*prog, 1.7 - 0.5*prog])
        # Le mani si appoggiano sulle ginocchia
        points[3] = np.column_stack([ 0.3*np.ones_like(t), y_base - 0.2*prog, 0.9 - 0.3*prog])
        points[4] = np.column_stack([-0.3*np.ones_like(t), y_base - 0.2*prog, 0.9 - 0.3*prog])
        # I piedi restano PIANTATI a terra (Z=0, Y=avanti al bacino)
        points[5] = np.column_stack([-0.2*np.ones_like(t), np.full_like(t, y_base - 0.3), np.zeros_like(t)])
        points[6] = np.column_stack([ 0.2*np.ones_like(t), np.full_like(t, y_base - 0.3), np.zeros_like(t)])

    elif action == "bending":
        # Piega in avanti e poi si rialza
        bend = np.sin(np.pi * t / duration_sec)
        y_base = 2.0
        
        # Il bacino indietreggia leggermente per bilanciare, Z resta quasi uguale
        points[0] = np.column_stack([np.zeros_like(t), y_base + 0.2*bend, 1.0 - 0.1*bend])
        # Torso e testa si abbassano e vanno in avanti verso il radar
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
        # Soggetto disteso a terra con micro-respirazione sul torso
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
        drink = np.sin(np.pi * t / duration_sec)
        
        points[0] = np.column_stack([np.zeros_like(t), np.full_like(t, y_base), np.full_like(t, 1.0)])
        points[1] = np.column_stack([np.zeros_like(t), np.full_like(t, y_base), np.full_like(t, 1.4)])
        points[2] = np.column_stack([np.zeros_like(t), np.full_like(t, y_base), np.full_like(t, 1.7)])
        points[3] = np.column_stack([0.2*(1-drink), y_base - 0.2*drink, 0.9 + 0.7*drink]) # Mano DX va alla bocca
        points[4] = np.column_stack([-0.3*np.ones_like(t), np.full_like(t, y_base), np.full_like(t, 0.9)])
        points[5] = np.column_stack([-0.2*np.ones_like(t), np.full_like(t, y_base), np.zeros_like(t)])
        points[6] = np.column_stack([ 0.2*np.ones_like(t), np.full_like(t, y_base), np.zeros_like(t)])

    return points, rcs, t
'''

'''
def visualize_mocap(action="sitting", duration_sec=2.5, fs=SWEEP_RATE):
    # By calling this function I have the 7 points trajectory (based on the movement) 
    # with the respective 3D movements and the RCS weight vector for the 7 joints
    points, rcs, _ = generate_mocap_trajectory(action, duration_sec, fs)

    # Here I extract the total number of points that constitute the interval of simulation
    num_samples = points.shape[1]

    # Collegamenti anatomici corretti dal nuovo "Root" (Bacino=0)
    # Here I define the "stick-figure" of the skeleton as a list of indexes (i,j)
    # where the joints are connected one each other as described in the following
    limbs = [
        (0, 1),  # Pelvis -> Torso
        (1, 2),  # Torso -> Head
        (1, 3),  # Torso -> RX Hand
        (1, 4),  # Torso -> LX Hand
        (0, 5),  # Pelvis -> LX Foot
        (0, 6)   # Pelvis -> RX Foot
    ]
    ###############################
    # 3D SCENE CONFIGURATION
    ###############################

    # A graphical window is created and the axes are set with a three-dimensional space (projection="3d")
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # A Black marker is put at (0,0,1) and it's the radar fixed position
    ax.scatter([0], [0], [1], color="black", marker="^", s=150, label="Radar")

    ################
    # INITIAL BODY DRAWING
    ################

    # It draws the 7 points attached to the initial frame (frame = 0, or initial configuration), 
    # all the joints have a size of 400 weighted by their respective RCS
    scatter = ax.scatter(
        points[:, 0, 0], 
        points[:, 0, 1], 
        points[:, 0, 2],
        c="red", 
        s=rcs * 400, 
        label="Joints"
    )

    # It creates 7 empty 3D objects (one for every limb). They will be updated dinamically for every frame
    lines = [ax.plot([], [], [], color="blue", linewidth=3)[0] for _ in limbs]

    ##############################
    # VIEW AND AXIS CONFIGURATION
    ##############################

    # Blocks the min and mx limits of the 3 axes. By blocking those, matplotlib won't
    # resize or move the camera during the object movement
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([0.0, 3.5])
    ax.set_zlim([0.0, 2.0])

    # Labels for the axis coordinates and the dynamic titles based on the performed actions
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y (Distance from the radar) [m]")
    ax.set_zlabel("Z (Height) [m]")
    ax.set_title(f"3D Motion Visualization for the action: {action.upper()}")
    ax.legend(loc="upper right")
    '''
    This function is called from FuncAnimation at every step to update the trajectory and make the animation
    '''
    def update(frame):
        # Here it extracts the (x,y,z) coordinates of all 7 points of the body at the current frame (num_samples)
        pos = points[:, frame, :]
        # It updates the 3D space positions of the 7 red joints without re-drawing the entire scene from scratch 
        scatter._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])

        # It cycles on every limb and updates the line that unites point i and point j:
        # - set_data --> updates the x,y components
        # - set_3d_properties --> updates the z component
        for line, (i, j) in zip(lines, limbs):
            line.set_data([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]])
            line.set_3d_properties([pos[i, 2], pos[j, 2]])

        # An updates list of all the graphical components modified in the frame is returned
        return [scatter] + lines

    # Initialization of the animation object, where: 'fig' is the reference figure, 
    # 'update' is the function to be called at every frame, 'frame' is the total 
    # number of frames to show, 'interval' it's the delay between two frames 
    # (with fs=10Hz we have an interval of 10ms, which is realistic), 'blit=False' 
    # necessary for 3D plots in matplotlib, 'repeat=True' lets the motion to start 
    # from the beginning once it's finshed.
    anim = FuncAnimation(fig, update, 
                         frames=num_samples, 
                         interval=1000/fs, 
                         blit=False, 
                         repeat=True)
    plt.show()

    # Saving the animation as a .gif file
    #anim.save(f"{ACTIONS[5]}_stickman.gif",fps=60)

if __name__ == "__main__":
    # Prova "sitting", "bending" o "falling" per vedere i piedi piantati a terra
    visualize_mocap(action=ACTIONS[3])