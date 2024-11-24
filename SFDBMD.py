import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Set up the Civil Engineering theme
sns.set_theme(style="whitegrid")

# Suppress divide by zero warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------------
# Function to switch between Load Cases
# ----------------------------
def get_train_loads(load_case, locomotive_at_front=True):
    if load_case == 1:
        # Load Case 1: Total weight 400 N, equally distributed
        P_total = 400  # Total train weight in N
        axle_loads = np.full(6, P_total / 6)
        # Axle offsets in mm from the start of the train
        axle_offsets = np.array([52, 228, 392, 568, 732, 908])
        train_length = 960  # mm
    elif load_case == 2:
        # Load Case 2: Base case example with locomotive and freight cars
        # Freight cars and locomotive weights
        weight_freight_car_light = 135  # N
        weight_freight_car_heavy = 1.10 * weight_freight_car_light  # N
        weight_locomotive = 1.35 * weight_freight_car_heavy  # N
        # Axle loads depend on positions
        # Assume locomotive at the front or end
        if locomotive_at_front:
            # Locomotive at front
            axle_loads = np.array([
                weight_locomotive / 2, weight_locomotive / 2,
                weight_freight_car_heavy /2, weight_freight_car_heavy /2,
                weight_freight_car_light /2, weight_freight_car_light /2,
            ])
        else:
            # Locomotive at end
            axle_loads = np.array([
                weight_freight_car_light /2, weight_freight_car_light /2,
                weight_freight_car_heavy /2, weight_freight_car_heavy /2,
                weight_locomotive / 2, weight_locomotive / 2,
            ])
        P_total = np.sum(axle_loads)
        # Axle offsets in mm from the start of the train
        # Car lengths (assumed)
        locomotive_length = 200  # mm (assumed)
        freight_car_length = 200  # mm (assumed)
        # Axle positions based on assumed lengths and spacings
        axle_offsets = np.array([
            0 + locomotive_length / 4, locomotive_length * 3 / 4,  # Locomotive axles
            locomotive_length + freight_car_length / 4, locomotive_length + freight_car_length * 3 / 4,  # Heavy freight car axles
            locomotive_length + freight_car_length + freight_car_length / 4,
            locomotive_length + freight_car_length + freight_car_length * 3 / 4  # Light freight car axles
        ])
        train_length = locomotive_length + 2 * freight_car_length  # mm
    else:
        raise ValueError("Invalid load case. Choose 1 or 2.")
    return P_total, axle_loads, axle_offsets, train_length

# ----------------------------
# Function to compute SFD and BMD at specific train positions
# ----------------------------
def compute_SFD_BMD(load_case, design_params, position, locomotive_at_front=True):
    # Initialize Parameters
    L = 1200  # Total length of the bridge in mm
    n = 1200  # Number of segments (1 mm per segment)
    x = np.linspace(0, L, n+1)  # x-axis positions along the bridge

    # Get train loads and axle positions
    P_total, axle_loads, axle_offsets, train_length = get_train_loads(load_case, locomotive_at_front)

    # Place train at specified position (position of front of train)
    xp = position  # Position of the front of the train on the bridge
    axles = xp + axle_offsets

    # Only consider axles that are on the bridge
    on_bridge = (axles >= 0) & (axles <= L)
    axles_on_bridge = axles[on_bridge]
    axle_loads_on_bridge = axle_loads[on_bridge]

    # Reaction forces using equilibrium equations
    total_load = np.sum(axle_loads_on_bridge)
    moments = np.sum(axle_loads_on_bridge * axles_on_bridge)
    R2 = moments / L
    R1 = total_load - R2

    # Shear force and bending moment arrays
    V = np.zeros_like(x)
    M = np.zeros_like(x)

    # Calculate shear force and bending moment at each point
    for i in range(len(x)):
        xi = x[i]
        V[i] = R1
        M[i] = R1 * xi

        # Subtract loads from axles to the left of xi
        loads_left = axle_loads_on_bridge[axles_on_bridge <= xi]
        positions_left = axles_on_bridge[axles_on_bridge <= xi]
        V[i] -= np.sum(loads_left)
        M[i] -= np.sum(loads_left * (xi - positions_left))

    return x, V, M, R1, R2, axles_on_bridge

# ----------------------------
# Function to find train positions causing max shear and moment
# ----------------------------
def find_max_shear_moment_positions(load_case, design_params, locomotive_at_front=True):
    L = 1200  # Length of bridge in mm
    P_total, axle_loads, axle_offsets, train_length = get_train_loads(load_case, locomotive_at_front)

    # For Load Case 1, positions are predefined
    if load_case == 1:
        # Maximum bending moment occurs when train is centered
        position_max_moment = (L - train_length) / 2  # Position of train front
        x_max_moment = L / 2  # Mid-span
        # Maximum shear occurs when train is just entering the bridge
        position_max_shear = 0  # Train starting at the beginning
        x_max_shear = 0  # Shear is maximum at support
    else:
        # For Load Case 2, need to find positions
        # We will loop over possible positions
        n_positions = L + 1  # Move train in 1 mm increments
        train_positions = np.linspace(0, L - train_length, n_positions)
        max_shear = 0
        max_moment = 0
        position_max_shear = 0
        position_max_moment = 0
        x_max_shear = 0
        x_max_moment = 0

        for xp in train_positions:
            x_current, V, M, R1, R2, axles_on_bridge = compute_SFD_BMD(load_case, design_params, xp, locomotive_at_front)
            V_abs_max = np.max(np.abs(V))
            idx_V_max = np.argmax(np.abs(V))
            if V_abs_max > np.abs(max_shear):
                max_shear = V[idx_V_max]
                x_max_shear = x_current[idx_V_max]
                position_max_shear = xp

            M_abs_max = np.max(np.abs(M))
            idx_M_max = np.argmax(np.abs(M))
            if M_abs_max > np.abs(max_moment):
                max_moment = M[idx_M_max]
                x_max_moment = x_current[idx_M_max]
                position_max_moment = xp

    return position_max_shear, position_max_moment, x_max_shear, x_max_moment

# ----------------------------
# Plotting Function for SFD and BMD
# ----------------------------
def plot_SFD_BMD_specific(x, V, M, load_case, design_name, position, max_value_info, x_max_shear, x_max_moment, axles_on_bridge):
    # Invert y-axis of BMD plot
    invert_bmd_yaxis = True

    # Plot SFD and BMD
    plt.figure(figsize=(12, 6))

    # Shear Force Diagram
    plt.subplot(1, 2, 1)
    plt.step(x, V, 'b', where='post', label='Shear Force Diagram')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.title(f'SFD\n{design_name}, Load Case {load_case}\nTrain front at x = {position:.1f} mm')
    plt.xlabel('Position along bridge (mm)')
    plt.ylabel('Shear Force (N)')
    plt.legend()
    plt.grid(True)
    plt.text(0.05, 0.95, max_value_info.get('shear_text', ''), transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top')
    # Highlight the point of maximum shear
    plt.plot(x_max_shear, V[np.where(x == x_max_shear)], 'ro')
    # Mark axle positions
    for axle in axles_on_bridge:
        plt.axvline(axle, color='grey', linestyle='--', linewidth=0.5)

    # Bending Moment Diagram
    plt.subplot(1, 2, 2)
    plt.plot(x, M, 'g', label='Bending Moment Diagram')
    if invert_bmd_yaxis:
        plt.gca().invert_yaxis()
    plt.axhline(0, color='black', linewidth=0.5)
    plt.title(f'BMD\n{design_name}, Load Case {load_case}\nTrain front at x = {position:.1f} mm')
    plt.xlabel('Position along bridge (mm)')
    plt.ylabel('Bending Moment (N·mm)')
    plt.legend()
    plt.grid(True)
    plt.text(0.05, 0.95, max_value_info.get('moment_text', ''), transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top')
    # Highlight the point of maximum moment
    plt.plot(x_max_moment, M[np.where(x == x_max_moment)], 'ro')
    # Mark axle positions
    for axle in axles_on_bridge:
        plt.axvline(axle, color='grey', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

# ----------------------------
# Define Designs
# ----------------------------
designs = {
    'Design 0': {
        'design_name': 'Design 0',
        'b_top': 100,  # mm
        't_top': 1.27,  # mm
        'b_bottom': 80,  # mm
        't_bottom': 1.27,  # mm
        'h': 75,  # mm
        't_web': 1.27,  # mm
        's_web': 70,  # mm
        'diaphragm_positions': [0, 400, 800, 1200],  # mm
    }
}

# ----------------------------
# Main Execution
# ----------------------------
for load_case in [1, 2]:
    design = designs['Design 0']
    design_name = design['design_name']
    locomotive_at_front = True  # You can change this to False to place locomotive at the end

    # Find positions causing maximum shear and moment
    position_max_shear, position_max_moment, x_max_shear, x_max_moment = find_max_shear_moment_positions(
        load_case, design, locomotive_at_front)

    # Compute SFD and BMD at position of maximum shear
    x_shear, V_shear, M_shear, R1_shear, R2_shear, axles_shear = compute_SFD_BMD(
        load_case, design, position_max_shear, locomotive_at_front)
    max_shear_value = V_shear[np.argmax(np.abs(V_shear))]
    x_shear_value = x_shear[np.argmax(np.abs(V_shear))]

    # Prepare info for plotting
    max_shear_info = {
        'shear_text': f'Max Shear Force = {max_shear_value:.2f} N at x = {x_shear_value:.1f} mm',
    }

    # Plot SFD and BMD at position of maximum shear
    plot_SFD_BMD_specific(x_shear, V_shear, M_shear, load_case, design_name,
                          position_max_shear, max_shear_info, x_shear_value, x_max_moment, axles_shear)

    # Compute SFD and BMD at position of maximum moment
    x_moment, V_moment, M_moment, R1_moment, R2_moment, axles_moment = compute_SFD_BMD(
        load_case, design, position_max_moment, locomotive_at_front)
    max_moment_value = M_moment[np.argmax(np.abs(M_moment))]
    x_moment_value = x_moment[np.argmax(np.abs(M_moment))]

    # Prepare info for plotting
    max_moment_info = {
        'moment_text': f'Max Bending Moment = {max_moment_value:.2f} N·mm at x = {x_moment_value:.1f} mm',
    }

    # Plot SFD and BMD at position of maximum moment
    plot_SFD_BMD_specific(x_moment, V_moment, M_moment, load_case, design_name,
                          position_max_moment, max_moment_info, x_shear_value, x_moment_value, axles_moment)