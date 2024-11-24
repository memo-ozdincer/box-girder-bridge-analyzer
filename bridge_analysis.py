import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
# ----------------------------
# Set up the Civil Engineering theme
# ----------------------------
sns.set_theme(style="whitegrid")

# Suppress divide by zero warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------------
# Function to switch between Load Cases
# ----------------------------
def get_train_loads(load_case):
    if load_case == 1:
        # Load Case 1: Total weight 400 N, equally distributed
        P_total = 400  # Total train weight in N
        axle_loads = np.full(6, P_total / 6)
    elif load_case == 2:
        # Load Case 2: Base case example with locomotive and freight cars
        weight_freight_car = 135  # N
        weight_locomotive = 1.35 * weight_freight_car  # N
        axle_loads = np.array([weight_locomotive / 2, weight_locomotive / 2,
                               weight_freight_car / 2, weight_freight_car / 2,
                               weight_freight_car / 2, weight_freight_car / 2])
        P_total = np.sum(axle_loads)
    else:
        raise ValueError("Invalid load case. Choose 1 or 2.")
    return P_total, axle_loads

# ----------------------------
# Function to define Bridge Geometry
# ----------------------------
def define_geometry(design_params, x):
    # Extract parameters from the design dictionary
    b_top = np.full_like(x, design_params['b_top'])  # mm
    t_top = np.full_like(x, design_params['t_top'])  # mm
    b_bottom = np.full_like(x, design_params['b_bottom'])  # mm
    t_bottom = np.full_like(x, design_params['t_bottom'])  # mm
    h_web = np.full_like(x, design_params['h']) - t_top - t_bottom  # mm
    t_web = np.full_like(x, design_params['t_web'])  # mm
    s_web = np.full_like(x, design_params['s_web'])  # mm (web spacing)
    diaphragm_positions = np.array(design_params['diaphragm_positions'])  # mm

    # Apply any thickness variations
    if 'thickness_variation' in design_params:
        for var in design_params['thickness_variation']:
            idx = (x >= var['start']) & (x <= var['end'])
            t_top[idx] = var.get('t_top', t_top[idx])
            t_bottom[idx] = var.get('t_bottom', t_bottom[idx])
            t_web[idx] = var.get('t_web', t_web[idx])
            b_top[idx] = var.get('b_top', b_top[idx])
            s_web[idx] = var.get('s_web', s_web[idx])
    return b_top, t_top, b_bottom, t_bottom, h_web, t_web, s_web, diaphragm_positions

# ----------------------------
# Main Analysis Function
# ----------------------------
def bridge_analysis(load_case, design_params):
    # Initialize Parameters
    L = design_params['L']  # Use design-specific length
    n = int(L)  # Number of segments (1 mm per segment)
    x = np.linspace(0, L, n+1)  # x-axis positions along the bridge

    # Material properties
    E_m = 4000  # Young's Modulus in MPa
    nu_m = 0.2  # Poisson's ratio
    sigma_t_ult = 30  # Tensile strength in MPa
    sigma_c_ult = 6  # Compressive strength in MPa
    tau_m_ult = 4  # Shear strength of matboard in MPa
    tau_g_ult = 2  # Shear strength of glue in MPa

    # Get train loads
    P_total, axle_loads = get_train_loads(load_case)
    axle_loads = axle_loads  # N per axle

    # Train dimensions and axle positions
    train_length = 960  # Total length of train in mm
    axle_offsets = np.array([52, 228, 392, 568, 732, 908])  # Updated axle positions

    # Define Bridge Geometry
    b_top, t_top, b_bottom, t_bottom, h_web, t_web, s_web, diaphragm_positions = define_geometry(design_params, x)

    # Initialize arrays
    A_total = np.zeros_like(x)
    y_bar = np.zeros_like(x)
    I = np.zeros_like(x)
    Q_cent = np.zeros_like(x)
    Q_glue = np.zeros_like(x)
    c_top = np.zeros_like(x)
    c_bottom = np.zeros_like(x)

    for i in range(len(x)):
        # Areas of components
        A_top = b_top[i] * t_top[i]
        A_bottom = b_bottom[i] * t_bottom[i]
        A_web = 2 * t_web[i] * h_web[i]

        # Distance from bottom to centroid of each component
        y_top = t_bottom[i] + h_web[i] + t_top[i] / 2
        y_bottom = t_bottom[i] / 2
        y_web = t_bottom[i] + h_web[i] / 2

        # Total area
        A_total[i] = A_top + A_bottom + A_web

        # Centroid calculation
        y_bar[i] = (A_top * y_top + A_bottom * y_bottom + A_web * y_web) / A_total[i]

        # Moment of inertia calculation
        I_top = (b_top[i] * t_top[i] ** 3) / 12 + A_top * (y_top - y_bar[i]) ** 2
        I_bottom = (b_bottom[i] * t_bottom[i] ** 3) / 12 + A_bottom * (y_bottom - y_bar[i]) ** 2
        I_webs = 2 * ((t_web[i] * h_web[i] ** 3) / 12 + t_web[i] * h_web[i] * (y_web - y_bar[i]) ** 2)
        I[i] = I_top + I_bottom + I_webs

        # First moment of area Q at neutral axis for shear stress
        A_above_na = A_top + t_web[i] * (h_web[i] / 2)
        y_above_na = (A_top * y_top + t_web[i] * (h_web[i] / 2) * y_web) / A_above_na
        Q_cent[i] = A_above_na * (y_above_na - y_bar[i])

        # Q at glue interface (assuming glue between top flange and webs)
        Q_glue[i] = A_top * (y_top - y_bar[i])

        # c_top and c_bottom
        c_top[i] = y_top - y_bar[i]
        c_bottom[i] = y_bar[i] - y_bottom

    # Train Movement and Load Effects
    V_env = np.zeros_like(x)
    M_env = np.zeros_like(x)

    train_positions = np.arange(0, L - train_length + 1, 1)  # Train positions at every mm along bridge

    for xp in train_positions:
        # Axle positions along the bridge
        axles = xp + axle_offsets

        # Reaction forces using equilibrium equations
        total_load = np.sum(axle_loads)
        moments = np.sum(axle_loads * axles)
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
            loads_left = axle_loads[axles <= xi]
            positions_left = axles[axles <= xi]
            V[i] -= np.sum(loads_left)
            M[i] -= np.sum(loads_left * (xi - positions_left))

        # Update envelopes
        V_env = np.maximum(V_env, np.abs(V))
        M_env = np.maximum(M_env, np.abs(M))

    # Modify V_env to show positive to negative transition
    V_env = V_env * np.sign(R1)

    # Calculate Applied Stresses
    sigma_top = M_env * c_top / I  # Compressive stress at top flange
    sigma_bottom = M_env * c_bottom / I  # Tensile stress at bottom flange

    # Shear stress in webs at neutral axis
    b_web = 2 * t_web  # Total web thickness for shear
    tau_web = V_env * Q_cent / (I * b_web)  # Shear stress in webs

    # Shear stress in glue interfaces
    tau_glue = V_env * Q_glue / (I * b_top)

    # Calculate Failure Capacities
    # Material capacities
    sigma_tension_capacity = np.full_like(x, sigma_t_ult)
    sigma_compression_capacity = np.full_like(x, sigma_c_ult)
    tau_shear_capacity = np.full_like(x, tau_m_ult)
    tau_glue_capacity = np.full_like(x, tau_g_ult)

    # Plate buckling capacities
    # Failure Mode 5: Buckling of compressive flange between webs
    k1 = 4  # Buckling coefficient for flange between webs
    b_in = s_web - t_web  # Unsupported flange width between webs
    sigma_buckling_flange = (k1 * np.pi ** 2 * E_m) / (12 * (1 - nu_m ** 2)) * (t_top / b_in) ** 2  # Varies with x

    # Failure Mode 6: Buckling of flange tips
    b_out = (b_top - s_web) / 2  # Unsupported flange width outside webs
    k2 = 0.425  # Buckling coefficient for flange tips
    sigma_buckling_flange_tips = (k2 * np.pi ** 2 * E_m) / (12 * (1 - nu_m ** 2)) * (t_top / b_out) ** 2

    # Failure Mode 7: Buckling of webs due to flexural stresses
    k3 = 6
    sigma_buckling_webs = (k3 * np.pi ** 2 * E_m) / (12 * (1 - nu_m ** 2)) * (t_web / h_web) ** 2

    # Failure Mode 8: Shear buckling of webs
    diaphragm_intervals = np.diff(diaphragm_positions)
    a_avg = np.mean(diaphragm_intervals) if len(diaphragm_intervals) > 0 else L
    tau_buckling_webs = (5 * np.pi ** 2 * E_m) / (12 * (1 - nu_m ** 2)) * \
                        ((t_web / h_web) ** 2 + (t_web / a_avg) ** 2)

    # Factors of Safety
    FOS_tension = np.where(sigma_bottom != 0,
                           sigma_tension_capacity / sigma_bottom, np.inf)
    FOS_compression = np.where(sigma_top != 0,
                               sigma_compression_capacity / sigma_top, np.inf)
    FOS_shear = np.where(tau_web != 0, tau_shear_capacity / tau_web, np.inf)
    FOS_glue = np.where(tau_glue != 0, tau_glue_capacity / tau_glue, np.inf)
    FOS_buckling_flange = np.where(sigma_top != 0,
                                   sigma_buckling_flange / sigma_top, np.inf)
    FOS_buckling_flange_tips = np.where(sigma_top != 0,
                                        sigma_buckling_flange_tips / sigma_top, np.inf)
    FOS_buckling_webs = np.where(sigma_top != 0,
                                 sigma_buckling_webs / sigma_top, np.inf)
    FOS_shear_buckling = np.where(tau_web != 0,
                                  tau_buckling_webs / tau_web, np.inf)

    # Find minimum FOS across all failure modes
    FOS_all = np.vstack([FOS_tension, FOS_compression, FOS_shear, FOS_glue,
                         FOS_buckling_flange, FOS_buckling_flange_tips,
                         FOS_buckling_webs, FOS_shear_buckling])

    min_FOS = np.min(FOS_all, axis=0)
    overall_min_FOS = np.min(min_FOS)
    critical_location = x[np.argmin(min_FOS)]

    # Determine controlling failure mode at each point
    failure_modes = ['Tension', 'Compression', 'Shear', 'Glue Shear',
                     'Flange Buckling (Between Webs)', 'Flange Buckling (Tips)',
                     'Web Buckling', 'Shear Buckling']
    controlling_mode = np.argmin(FOS_all, axis=0)
    critical_failure_mode = failure_modes[np.argmin(FOS_all[:, np.argmin(min_FOS)])]

    # Adjust the applied load proportionally to the minimum FOS to find the maximum load
    P_fail = P_total * overall_min_FOS

    # Output results
    print("\n=== Bridge Analysis Results ===")
    print(f"Load Case: {load_case}")
    print(f"Design Choice: {design_params['design_name']}")
    print(f"Minimum Factor of Safety: {overall_min_FOS:.2f} at x = {critical_location:.1f} mm")
    print(f"Controlling Failure Mode: {critical_failure_mode}")
    print(f"Predicted Failure Load: {P_fail:.2f} N")

    # Shear capacity calculations
    V_fail_shear = (tau_shear_capacity * I * b_web) / Q_cent  # Shear failure capacity in N
    V_fail_glue = (tau_glue_capacity * I * b_top) / Q_glue  # Glue shear failure capacity in N
    V_fail_shear_buckling = (tau_buckling_webs * I * b_web) / Q_cent  # Shear buckling capacity in N

    # Bending moment capacities
    M_fail_tension = sigma_tension_capacity * I / c_bottom  # Tensile failure capacity in N·mm
    M_fail_compression = sigma_compression_capacity * I / c_top  # Compressive failure capacity in N·mm
    M_fail_buckling_flange = sigma_buckling_flange * I / c_top  # Flange buckling capacity in N·mm
    M_fail_buckling_flange_tips = sigma_buckling_flange_tips * I / c_top  # Flange tip buckling capacity in N·mm
    M_fail_buckling_webs = sigma_buckling_webs * I / (h_web / 2)  # Web buckling capacity in N·mm

    return x, V_env, M_env, V_fail_shear, V_fail_glue, V_fail_shear_buckling, \
           M_fail_tension, M_fail_compression, M_fail_buckling_flange, M_fail_buckling_flange_tips, M_fail_buckling_webs, \
           failure_modes, FOS_all, controlling_mode
def calculate_material_usage(design_params):
    # Get bridge length
    L = design_params['L']
    
    # Basic dimensions
    b_top = design_params['b_top']
    t_top = design_params['t_top']
    b_bottom = design_params['b_bottom']
    t_bottom = design_params['t_bottom']
    h = design_params['h']
    t_web = design_params['t_web']
    s_web = design_params['s_web']
    
    # Calculate areas
    # Top flange area
    top_flange_area = b_top * L
    
    # Bottom flange area
    bottom_flange_area = b_bottom * L
    
    # Web areas (2 webs)
    web_height = h - t_top - t_bottom
    web_area = 2 * web_height * L
    
    # Diaphragm areas
    diaphragm_positions = design_params['diaphragm_positions']
    num_diaphragms = len(diaphragm_positions)
    diaphragm_area = num_diaphragms * (h - t_top - t_bottom) * s_web
    
    # Account for thickness variations if they exist
    if 'thickness_variation' in design_params:
        for var in design_params['thickness_variation']:
            length = var['end'] - var['start']
            if 't_top' in var:
                top_flange_area += b_top * length * (var['t_top'] - t_top)
            if 't_bottom' in var:
                bottom_flange_area += b_bottom * length * (var['t_bottom'] - t_bottom)
            if 't_web' in var:
                web_area += 2 * web_height * length * (var['t_web'] - t_web)
    
    # Total area in mm²
    total_area = top_flange_area + bottom_flange_area + web_area + diaphragm_area
    
    # Convert to m²
    total_area_m2 = total_area / 1e6
    
    return {
        'total_area_mm2': total_area,
        'total_area_m2': total_area_m2,
        'components': {
            'top_flange_mm2': top_flange_area,
            'bottom_flange_mm2': bottom_flange_area,
            'webs_mm2': web_area,
            'diaphragms_mm2': diaphragm_area
        }
    }
def set_civil_engineering_style():
    # Custom color palette inspired by construction materials
    colors = {
        'timber_brown': '#8B4513',
        'steel_grey': '#71797E',
        'concrete_beige': '#E8DCC4',
        'rust_orange': '#A0522D',
        'blueprint_blue': '#1E4D6B',
        'warning_red': '#8B0000'
    }
    
    # Set the style
    plt.style.use('seaborn-v0_8-paper')
    
    # Custom parameters
    plt.rcParams.update({
        'figure.facecolor': colors['concrete_beige'],
        'axes.facecolor': '#F5F5DC',  # Antique white
        'axes.edgecolor': colors['timber_brown'],
        'axes.labelcolor': colors['timber_brown'],
        'xtick.color': colors['timber_brown'],
        'ytick.color': colors['timber_brown'],
        'grid.color': '#BC8F8F',  # Rosybrown
        'grid.linestyle': '--',
        'grid.alpha': 0.3,
        'text.color': colors['timber_brown'],
        'figure.titlesize': 16,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'font.family': 'serif'
    })
    
    return colors
def plot_failure_envelopes(x, V_env, V_fail_shear, V_fail_glue, V_fail_shear_buckling,
                          M_env, M_fail_tension, M_fail_compression, M_fail_buckling_flange,
                          M_fail_buckling_flange_tips, M_fail_buckling_webs, load_case, design_name):
    
    colors = set_civil_engineering_style()
    
    # Add more engineering-themed colors
    colors.update({
        'parchment': '#F5E6D3',
        'bridge_steel': '#708090',
        'safety_orange': '#FF5733',
        'stress_red': '#B22222',
        'tension_blue': '#4682B4',
        'wood_grain': '#DEB887'
    })
    
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(colors['concrete_beige'])
    
    # Add texture to the background (subtle grid pattern)
    for i in range(0, 1800, 50):
        for j in range(0, 1200, 50):
            plt.plot([i, i+30], [j, j+30], color=colors['wood_grain'], alpha=0.1, linewidth=0.5)
    
    subplot_params = dict(
        facecolor=colors['parchment'],
        edgecolor=colors['timber_brown'],
        linewidth=2
    )

    def style_subplot(ax, title, xlabel, ylabel):
        ax.set_facecolor(subplot_params['facecolor'])
        ax.set_title(title, pad=20, color=colors['timber_brown'], fontweight='bold')
        ax.set_xlabel(xlabel, color=colors['timber_brown'])
        ax.set_ylabel(ylabel, color=colors['timber_brown'])
        ax.grid(True, linestyle='--', alpha=0.3, color=colors['timber_brown'])
        ax.tick_params(colors=colors['timber_brown'])
        
        # Add corner decorations (fixed positioning)
        for corner in ['top-left', 'top-right', 'bottom-left', 'bottom-right']:
            x_pos = ax.get_xlim()[1] if 'right' in corner else ax.get_xlim()[0]
            y_pos = ax.get_ylim()[1] if 'top' in corner else ax.get_ylim()[0]
            ax.plot(x_pos, y_pos, 'o', color=colors['steel_grey'],
                   markersize=10, clip_on=False, zorder=10,
                   markeredgecolor=colors['bridge_steel'], markeredgewidth=2)

    # Shear Force Analysis plots
    ax1 = plt.subplot(2, 3, 1)
    plt.step(x, V_env, color=colors['blueprint_blue'], where='post', 
             label='Shear Force Envelope', linewidth=2)
    plt.plot(x, V_fail_shear, color=colors['stress_red'], 
             label='Shear Failure Capacity', linewidth=2)
    plt.axhline(0, color=colors['timber_brown'], linewidth=0.5)
    style_subplot(ax1, 'Matboard Shear Failure', 'Position along bridge (mm)', 'Shear Force (N)')
    ax1.legend(facecolor=colors['parchment'], framealpha=0.9, edgecolor=colors['timber_brown'])

    # Glue Shear Failure
    ax2 = plt.subplot(2, 3, 2)
    plt.yscale('symlog')
    plt.step(x, V_env, color=colors['blueprint_blue'], where='post', 
             label='Shear Force Envelope', linewidth=2)
    plt.plot(x, V_fail_glue, color=colors['stress_red'], 
             label='Glue Shear Failure Capacity', linewidth=2)
    plt.axhline(0, color=colors['timber_brown'], linewidth=0.5)
    style_subplot(ax2, 'Glue Shear Failure', 'Position along bridge (mm)', 'Shear Force (N)')
    ax2.legend(facecolor=colors['parchment'], framealpha=0.9, edgecolor=colors['timber_brown'])

    # Matboard Shear Buckling
    ax3 = plt.subplot(2, 3, 3)
    plt.step(x, V_env, color=colors['blueprint_blue'], where='post', 
             label='Shear Force Envelope', linewidth=2)
    plt.plot(x, V_fail_shear_buckling, color=colors['stress_red'], 
             label='Shear Buckling Capacity', linewidth=2)
    plt.axhline(0, color=colors['timber_brown'], linewidth=0.5)
    style_subplot(ax3, 'Matboard Shear Buckling', 'Position along bridge (mm)', 'Shear Force (N)')
    ax3.legend(facecolor=colors['parchment'], framealpha=0.9, edgecolor=colors['timber_brown'])

    # Material Bending Failure
    ax4 = plt.subplot(2, 3, 4)
    ax4.invert_yaxis()
    plt.plot(x, M_env, color=colors['blueprint_blue'], label='Bending Moment Envelope', linewidth=2)
    plt.plot(x, M_fail_tension, color=colors['stress_red'], label='Tension Failure Capacity', linewidth=2)
    plt.plot(x, M_fail_compression, color=colors['tension_blue'], label='Compression Failure Capacity', linewidth=2)
    plt.axhline(0, color=colors['timber_brown'], linewidth=0.5)
    style_subplot(ax4, 'Material Bending Failure', 'Position along bridge (mm)', 'Bending Moment (N·mm)')
    ax4.legend(facecolor=colors['parchment'], framealpha=0.9, edgecolor=colors['timber_brown'])

    # Flange Buckling Failure
    ax5 = plt.subplot(2, 3, 5)
    ax5.invert_yaxis()
    plt.plot(x, M_env, color=colors['blueprint_blue'], label='Bending Moment Envelope', linewidth=2)
    plt.plot(x, M_fail_buckling_flange, color=colors['stress_red'], 
             label='Flange Buckling Capacity', linewidth=2)
    plt.plot(x, M_fail_buckling_flange_tips, color=colors['tension_blue'], 
             label='Flange Tip Buckling Capacity', linewidth=2)
    plt.axhline(0, color=colors['timber_brown'], linewidth=0.5)
    style_subplot(ax5, 'Flange Buckling Failure', 'Position along bridge (mm)', 'Bending Moment (N·mm)')
    ax5.legend(facecolor=colors['parchment'], framealpha=0.9, edgecolor=colors['timber_brown'])

    # Web Buckling Failure
    ax6 = plt.subplot(2, 3, 6)
    ax6.invert_yaxis()
    plt.plot(x, M_env, color=colors['blueprint_blue'], label='Bending Moment Envelope', linewidth=2)
    plt.plot(x, M_fail_buckling_webs, color=colors['stress_red'], 
             label='Web Buckling Capacity', linewidth=2)
    plt.axhline(0, color=colors['timber_brown'], linewidth=0.5)
    style_subplot(ax6, 'Web Buckling Failure', 'Position along bridge (mm)', 'Bending Moment (N·mm)')
    ax6.legend(facecolor=colors['parchment'], framealpha=0.9, edgecolor=colors['timber_brown'])

    # Add blueprint-style border
    fig.patch.set_linewidth(3)
    fig.patch.set_edgecolor(colors['bridge_steel'])
    
    # Add title with decorative elements
    title_text = f'Bridge Analysis Results, Load Case {load_case}, {design_name}'
   # plt.suptitle(title_text, fontsize=12, color=colors['timber_brown'], 
      #           fontweight='bold', y=0.95, 
      #           bbox=dict(facecolor=colors['parchment'], 
       #                   edgecolor=colors['timber_brown'],
        #                  boxstyle='round,pad=1'))

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.show()

# ----------------------------
# Define Designs
# ----------------------------
designs = {
    'Design 0': {
        'design_name': 'Design 0',
        'L': 1200, # mm
        'b_top': 100,  # mm
        't_top': 1.27,  # mm
        'b_bottom': 80,  # mm
        't_bottom': 1.27,  # mm
        'h': 75,  # mm
        't_web': 1.27,  # mm
        's_web': 70,  # mm
        'diaphragm_positions': [0, 400, 800, 1200],  # mm
    },
    'Design 1': {
        'design_name': 'Design 1',
        'L': 1250,  # Total length for Design 1
        'b_top': 100,  # mm
        't_top': 1.27,  # mm
        'b_bottom': 80,  # mm
        't_bottom': 1.27,  # mm
        'h': 140,  # mm
        't_web': 1.27,  # mm
        's_web': 40,  # mm
        'diaphragm_positions': [0, 200, 400, 625, 875, 1050, 1250],  # mm
        'thickness_variation': [
            {'start': 525, 'end': 725, 't_top': 2.54},
            {'start': 0, 'end': 200, 't_web': 2.54},
            {'start': 1050, 'end': 1250, 't_web': 2.54},
        ],
    },
    'Design 3': {
        'design_name': 'Design 3',
        'L': 1250,  # Total length for Design 3
        'b_top': 120,  # mm
        't_top': 1.27,  # mm
        'b_bottom': 60,  # mm
        't_bottom': 1.27,  # mm
        'h': 140,  # mm
        't_web': 1.27,  # mm
        's_web': 30,  # mm
        'diaphragm_positions': [0, 150, 300, 450, 600, 750, 900, 1050, 1250],  # mm
        'thickness_variation': [
            {'start': 475, 'end': 775, 't_top': 2.54},
            {'start': 0, 'end': 250, 't_web': 2.54},
            {'start': 1000, 'end': 1250, 't_web': 2.54},
        ],
    },
    'Design 4': {
        'design_name': 'Design 4',
        'L': 1200, # mm
        'b_top': 100,  # mm
        't_top': 1.27,  # mm
        'b_bottom': 80,  # mm
        't_bottom': 1.27,  # mm
        'h': 75,  # mm
        't_web': 2.27,  # mm
        's_web': 70,  # mm
        'diaphragm_positions': [],  # mm
    },
     'Design 7': {
        'design_name': 'Design 7',
        'L': 1200,  # mm
        'b_top': 100,  # mm
        't_top': 1.27,  # mm
        'b_bottom': 100,  # mm
        't_bottom': 1.27,  # mm
        'h': 140,  # mm
        't_web': 1.97,  # mm
        's_web': 40,  # mm
        'glue_tab_width': 5,  # mm
        't_glue_tab': 1.27,   # mm
        'diaphragm_positions': [0, 180, 360, 600, 840, 1020, 1200],  # mm
        # Thickness variations for bottom flange removal and top flange splicing
        'thickness_variation': [
            # Bottom flange removed for first and last 15%
            {'start': 0, 'end': 180, 't_bottom': 0},
            {'start': 1020, 'end': 1200, 't_bottom': 0},
            # Top flange splice overlap between 870 and 1020 mm (double thickness)
            {'start': 870, 'end': 1020, 't_top': 2 * 1.27},
            # Reinforcement from end of splice, 690 mm long
            {'start': 1020 - 690, 'end': 1020, 't_top': 2 * 1.27}
        ],
    }
}

# ----------------------------
# Run Analysis for Designs
# ----------------------------
# For Design 0 under Load Cases 1 and 2

for load_case in [1, 2]:
    x, V_env, M_env, _, _, _, _, _, _, _, _, _, _, _ = bridge_analysis(load_case, designs['Design 0'])
    
# Run analysis for one design under both load cases
design_to_analyze = 'Design 0'  # or whichever design you want to analyze
material_usage = calculate_material_usage(designs[design_to_analyze])
print(f"\nMaterial Usage for {design_to_analyze}:")
print(f"Total area: {material_usage['total_area_m2']:.4f} m²")
print("\nBreakdown:")
for component, area in material_usage['components'].items():
    print(f"{component}: {area/1e6:.4f} m²")
    
for load_case in [1, 2]:
    x, V_env, M_env, V_fail_shear, V_fail_glue, V_fail_shear_buckling, \
    M_fail_tension, M_fail_compression, M_fail_buckling_flange, M_fail_buckling_flange_tips, \
    M_fail_buckling_webs, failure_modes, FOS_all, controlling_mode = bridge_analysis(load_case, designs[design_to_analyze])

    plot_failure_envelopes(x, V_env, V_fail_shear, V_fail_glue, V_fail_shear_buckling,
                          M_env, M_fail_tension, M_fail_compression, M_fail_buckling_flange,
                          M_fail_buckling_flange_tips, M_fail_buckling_webs,
                          load_case=load_case, design_name=designs[design_to_analyze]['design_name'])