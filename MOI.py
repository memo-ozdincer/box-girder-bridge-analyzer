import numpy as np

def calculate_moment_of_inertia(design_params, x):
    # Extract design parameters and initialize variables
    b_top = np.full_like(x, design_params['b_top'])
    t_top = np.full_like(x, design_params['t_top'])
    b_bottom = np.full_like(x, design_params['b_bottom'])
    t_bottom = np.full_like(x, design_params['t_bottom'])
    h_web = np.full_like(x, design_params['h']) - t_top - t_bottom
    t_web = np.full_like(x, design_params['t_web'])
    w_tab = np.full_like(x, 5)  # Width of glue tabs
    t_tab = np.full_like(x, 1.27)  # Thickness of glue tabs

    # Initialize arrays for moment of inertia and centroid
    I = np.zeros_like(x)
    y_bar = np.zeros_like(x)

    for i in range(len(x)):
        # Component areas
        A_top = b_top[i] * t_top[i]
        A_bottom = b_bottom[i] * t_bottom[i]
        A_web_single = t_web[i] * h_web[i]
        A_web = 2 * A_web_single
        A_tab = w_tab[i] * t_tab[i]  # Area of one glue tab
        A_tabs_total = 4 * A_tab  # Total area of glue tabs

        # Centroid distances
        y_top = t_bottom[i] + h_web[i] + t_top[i] / 2
        y_bottom = t_bottom[i] / 2
        y_web = t_bottom[i] + h_web[i] / 2
        y_tab_top = t_bottom[i] + h_web[i] + t_top[i] - t_tab[i] / 2
        y_tab_bottom = t_tab[i] / 2

        # Total area and centroid location
        A_total = A_top + A_bottom + A_web + A_tabs_total
        y_bar[i] = (
            A_top * y_top +
            A_bottom * y_bottom +
            A_web * y_web +
            2 * A_tab * y_tab_top +
            2 * A_tab * y_tab_bottom
        ) / A_total

        # Calculate moment of inertia for each component
        I_top = (b_top[i] * t_top[i]**3) / 12 + A_top * (y_top - y_bar[i])**2
        I_bottom = (b_bottom[i] * t_bottom[i]**3) / 12 + A_bottom * (y_bottom - y_bar[i])**2
        I_web_single = (t_web[i] * h_web[i]**3) / 12
        I_webs = 2 * (I_web_single + A_web_single * (y_web - y_bar[i])**2)
        I_tab_local = (w_tab[i] * t_tab[i]**3) / 12
        I_tabs = 2 * (
            I_tab_local + A_tab * (y_tab_top - y_bar[i])**2
        ) + 2 * (
            I_tab_local + A_tab * (y_tab_bottom - y_bar[i])**2
        )

        # Total moment of inertia
        I[i] = I_top + I_bottom + I_webs + I_tabs

    return I

# Define the design parameters
designs = {
    'Design 0': {
        'b_top': 100,      # mm
        't_top': 1.27,     # mm
        'b_bottom': 80,    # mm
        't_bottom': 1.27,  # mm
        'h': 75,           # mm
        't_web': 1.27,     # mm,
    },
}

# Define the x-axis positions along the bridge
L = 1200  # Length of the bridge in mm
n = L     # Number of divisions (equal to the length here)
x = np.linspace(0, L, n+1)  # x-axis positions along the bridge

# Calculate moment of inertia
I = calculate_moment_of_inertia(designs['Design 0'], x)

# Print the moment of inertia at all positions
print("Moment of Inertia at each section (in mm^4):")
print(I)