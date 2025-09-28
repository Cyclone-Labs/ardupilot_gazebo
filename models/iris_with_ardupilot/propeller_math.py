import numpy as np
import math
from scipy.optimize import least_squares
import torch

# <air_density>: Density of the fluid this model is suspended in.
# <area>: Surface area of the link.
# <a0>: The initial "alpha" or initial angle of attack. a0 is also the y-intercept of the alpha-lift coefficient curve.
# <cla>: The ratio of the coefficient of lift and alpha slope before stall. Slope of the first portion of the alpha-lift coefficient curve.
# <cda>: The ratio of the coefficient of drag and alpha slope before stall.
# <cp>: Center of pressure. The forces due to lift and drag will be applied here.
# <forward>: 3-vector representing the forward direction of motion in the link frame.
# <upward>: 3-vector representing the direction of lift or drag.
# <alpha_stall>: Angle of attack at stall point; the peak angle of attack.
# <cla_stall>: The ratio of coefficient of lift and alpha slope after stall. Slope of the second portion of the alpha-lift coefficient curve.
# <cda_stall>: The ratio of coefficient of drag and alpha slope after stall.
# <control_joint_name>: Name of joint that actuates a control surface for this lifting body (Optional)
# <cm_delta>: How much Cm changes with a change in control surface deflection angle 

def calculate_propeller_forces(
    # Propeller geometry
    num_blades,               # Number of propeller blades
    blade_radius,            # m - Distance from hub center to blade center of pressure
    
    # Aerodynamic parameters
    air_density,             # kg/m³ - Density of the fluid
    area,                   # m² - Surface area of each blade
    a0,                     # rad - Initial angle of attack (alpha_0)
    cla,                    # 1/rad - Coefficient of lift vs alpha slope (before stall)
    cda,                    # 1/rad - Coefficient of drag vs alpha slope (before stall)
    alpha_stall,            # rad - Angle of attack at stall
    cla_stall,              # 1/rad - Coefficient of lift vs alpha slope (after stall)
    cda_stall,              # 1/rad - Coefficient of drag vs alpha slope (after stall)
    
    # Motion parameters
    angular_velocity,       # rad/s - Angular velocity of the propeller (scalar)
    
    # Optional parameters
    cma=0.0,                # 1/rad - Coefficient of moment vs alpha slope (before stall)
    cma_stall=0.0,          # 1/rad - Coefficient of moment vs alpha slope (after stall)
    min_velocity_threshold=0.01  # m/s - Minimum velocity threshold
):
    """
    Calculate aerodynamic forces on a propeller with multiple blades.
    
    Each blade is positioned radially around the hub, with:
    - Center of pressure at [blade_radius, 0, 0] in blade frame
    - Forward direction pointing tangentially (propeller direction)
    - Upward direction pointing radially outward
    
    Returns:
        dict: {
            'total_force': np.array([fx, fy, fz]),    # Total force from all blades (N)
            'total_torque': np.array([tx, ty, tz]),   # Total torque from all blades (N⋅m)
        }
    """
    
    # Check for valid inputs
    if angular_velocity == 0 or blade_radius == 0:
        return {
            'total_force': np.zeros(3),
            'total_torque': np.zeros(3),
        }
    
    total_force = np.zeros(3)
    total_torque = np.zeros(3)
    
    # Calculate forces for each blade
    for blade_idx in range(num_blades):
        # Blade angle around the hub
        blade_angle = 2 * math.pi * blade_idx / num_blades
        
        # Blade coordinate system:
        # - Center of pressure at distance blade_radius from hub
        # - Forward direction is tangential (propeller thrust direction)
        # - Upward direction is radially outward
        
        # Position of center of pressure (in hub frame)
        cp_x = blade_radius * math.cos(blade_angle)
        cp_y = blade_radius * math.sin(blade_angle)
        cp_z = 0.0
        cp = np.array([cp_x, cp_y, cp_z])
        
        # Forward direction (tangential, in direction of rotation)
        if angular_velocity > 0:
            forward = np.array([-math.sin(blade_angle), math.cos(blade_angle), 0.0])
        else:
            forward = np.array([math.sin(blade_angle), -math.cos(blade_angle), 0.0])
        
        # Upward direction (radially outward from hub)
        upward = np.array([0,0,1])
        
        # Velocity at center of pressure due to rotation
        # v = ω × r, where ω = [0, 0, angular_velocity]
        w = np.array([0, 0, angular_velocity])
        vel = np.cross(w, cp)
        vel_magnitude = np.linalg.norm(vel)
        
        # Check if velocity is above threshold
        if vel_magnitude <= min_velocity_threshold:
            continue
        
        # Normalize velocity vector
        vel_normalized = vel / vel_magnitude
        
        # Check if forward direction aligns with velocity
        if np.dot(forward, vel) <= 0.0:
            continue
        
        # Calculate spanwise direction (normal to lift-drag plane)
        spanwise = np.cross(forward, upward)
        spanwise = spanwise / np.linalg.norm(spanwise)
        
        # For a propeller blade, velocity is purely tangential, so no sweep correction needed
        cos2_sweep_angle = 1.0
        
        # Velocity is already in the lift-drag plane for a propeller
        vel_in_ld_plane = vel
        vel_ld_magnitude = vel_magnitude
        
        # Calculate drag direction (opposite to velocity)
        drag_direction = -vel_normalized
        
        # Calculate lift direction (perpendicular to drag in L-D plane)
        lift_direction = np.cross(spanwise, vel_in_ld_plane)
        lift_direction = lift_direction / np.linalg.norm(lift_direction)
        
        # Calculate angle of attack
        cos_alpha = np.clip(np.dot(lift_direction, upward), -1.0, 1.0)
        alpha = a0 - math.acos(cos_alpha)
        
        # Determine sign of alpha based on forward direction
        if np.dot(lift_direction, forward) >= 0.0:
            alpha = a0 + math.acos(cos_alpha)
        
        # Normalize alpha to [-π/2, π/2]
        while abs(alpha) > 0.5 * math.pi:
            alpha = alpha - math.pi if alpha > 0 else alpha + math.pi
        
        # Calculate dynamic pressure
        q = 0.5 * air_density * vel_ld_magnitude**2
        
        # Calculate coefficient of lift (cl)
        if alpha > alpha_stall:
            cl = (cla * alpha_stall + cla_stall * (alpha - alpha_stall)) * cos2_sweep_angle
            cl = max(0.0, cl)
        elif alpha < -alpha_stall:
            cl = (-cla * alpha_stall + cla_stall * (alpha + alpha_stall)) * cos2_sweep_angle
            cl = min(0.0, cl)
        else:
            cl = cla * alpha * cos2_sweep_angle
        
        # Calculate coefficient of drag (cd)
        if alpha > alpha_stall:
            cd = (cda * alpha_stall + cda_stall * (alpha - alpha_stall)) * cos2_sweep_angle
        elif alpha < -alpha_stall:
            cd = (-cda * alpha_stall + cda_stall * (alpha + alpha_stall)) * cos2_sweep_angle
        else:
            cd = cda * alpha * cos2_sweep_angle
        
        # Ensure drag is positive
        cd = abs(cd)
        
        # Calculate coefficient of moment (cm)
        if alpha > alpha_stall:
            cm = (cma * alpha_stall + cma_stall * (alpha - alpha_stall)) * cos2_sweep_angle
            cm = max(0.0, cm)
        elif alpha < -alpha_stall:
            cm = (-cma * alpha_stall + cma_stall * (alpha + alpha_stall)) * cos2_sweep_angle
            cm = min(0.0, cm)
        else:
            cm = cma * alpha * cos2_sweep_angle
        
        # Calculate forces
        lift_force = cl * q * area * lift_direction
        drag_force = cd * q * area * drag_direction
        moment_torque = cm * q * area * spanwise
        
        # Total force and torque for this blade
        blade_force = lift_force + drag_force
        blade_torque = moment_torque + np.cross(cp, blade_force)
        
        # Store blade results
        
        # Add to totals
        total_force += blade_force
        total_torque += blade_torque
    
    return (
        total_force[2],
        total_torque[2]
    )

import math

class ResidualsNetwork(torch.nn.Module):
    '''
    MLP with input shape (3,) and output shape (1,)
    2 hidden layers of 32 units each, ReLU activations
    '''
    def __init__(self):
        self.hidden_1 = torch.nn.Linear(3, 32)
        self.hidden_2 = torch.nn.Linear(32, 32)
        self.output = torch.nn.Linear(32, 1)
    
    def predict(self, x):
        # Simple linear model for demonstration
        x = torch.relu(self.hidden_1(x))
        x = torch.relu(self.hidden_2(x))
        return self.output(x)
    
    def update_from_flattened(self, params):
        # Update model parameters from a flattened array
        current_index = 0
        for param in self.parameters():
            param_length = param.numel()
            new_values = params[current_index:current_index + param_length].reshape(param.shape)
            param.data = torch.tensor(new_values, dtype=param.dtype)
            current_index += param_length

net = ResidualsNetwork()
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

# TODO: use this in motor plant and jointly optimize NN params

def run_motor_plant(throttle, amperage, load_torque, voltage=22.2, motor_kv=2100, motor_resistance=0.056, internal_friction = 0.001, motor_kt=0.00455, residual_parameters=np.zeros(22)):
    """
    Motor plant model including load torque dynamics
    
    throttle: [0,1] throttle input
    amperage: motor current (A)
    load_torque: external load torque (N⋅m)
    voltage: supply voltage (V)
    motor_kv: motor velocity constant (RPM/V)
    motor_resistance: motor resistance (Ω)
    motor_kt: motor torque constant (N⋅m/A) - if None, calculated from Kv
    
    Returns: angular velocity (rad/s)
    """
    applied_voltage = throttle * voltage
    
    # Calculate voltage drop across motor resistance
    resistive_drop = amperage * motor_resistance
    
    # Back EMF is the remaining voltage after resistive drop
    back_emf = applied_voltage - resistive_drop
    
    # Convert back EMF to no-load RPM using motor Kv
    no_load_rpm = max(0, back_emf * motor_kv)
    
    # Motor torque from current
    motor_torque = motor_kt * amperage
    
    # Simple approach: reduce RPM proportionally based on load
    # More sophisticated models would solve the full electrical-mechanical coupling
    
    # Method 1: Linear torque-speed relationship
    # Assume motor has some internal resistance to speed (like friction)
    
    # At steady state: motor_torque = load_torque + friction_torque
    # friction_torque = internal_friction * omega
    # So: omega = (motor_torque - load_torque) / internal_friction
    
    max_omega = no_load_rpm * 2 * math.pi / 60.0
    
    # Solve: motor_torque = load_torque + internal_friction * omega
    omega = max(0, (motor_torque - load_torque) / internal_friction)
    # Limit to no-load speed
    omega = min(omega, max_omega)
    
    return omega

def run_combined_plant(throttle, current, airfoil_params, no_load_current_constant):
    def calc_prop(omega):
        return calculate_propeller_forces(
            num_blades=3,
            air_density=1.2041,
            blade_radius=airfoil_params[0],
            area=airfoil_params[1],
            a0 = airfoil_params[2],
            cla = airfoil_params[3],
            cda = airfoil_params[4],
            cla_stall = airfoil_params[5],
            alpha_stall= airfoil_params[6],
            cda_stall = 0,
            angular_velocity=omega,
        )
    def fn(x):
        omega = run_motor_plant(throttle, current, x[0])
        force, torque = calc_prop(omega)
        return [torque - x[0]]
    steady_state_load_torque = least_squares(fn, [0.01], bounds=(0, np.inf), ftol=1e-12, xtol=1e-12).x[0]
    omega = run_motor_plant(throttle, current, steady_state_load_torque)
    force, torque = calc_prop(omega)
    return force

# Example usage
if __name__ == "__main__":
    data = np.array([ # (throttle, amps, grams thrust)
        (0.1, 2.3, 181),
        (0.2, 5.5, 393),
        (0.3, 9.9, 576),
        (0.4, 13.8, 576),
        (0.5, 17.9, 885),
        (0.6, 22.0, 1022),
        (0.7, 25.0, 1136),
        (0.8, 28.5, 1285),
        (0.9, 36.1, 1513),
        (1.0, 41.0, 1675)
    ])
    data[:,2] = data[:,2] / 1000.0 * 9.81 # convert grams to Newtons

    params_initial = np.array([
        0.03175, # 1.25 in to meters (blade center of pressure radius)
        0.0008, #area
        0.3, # a0
        4.25, # cla
        0.01, # cda
        0.025, # cla_stall
        1.4, # alpha_stall,
        1.4/10 # no load current per volt
    ])

    def make_predictions(params, data):
        res = []
        for row in data:
            throttle, measured_amps, measured_thrust = row
            predicted_thrust = run_combined_plant(throttle, measured_amps, params, params[7])
            res.append(predicted_thrust)
        return np.array(res)

    def residuals(params, data):
        res = []
        predictions = make_predictions(params, data)
        for i, row in enumerate(data):
            throttle, measured_amps, measured_thrust = row
            predicted_thrust = predictions[i]
            res.append(predicted_thrust-measured_thrust)
        return res

    initial_values = make_predictions(params_initial, data)
    print("Initial parameters:", params_initial)
    print("Initial predicted values:", initial_values)
    print("Actual values", data[:, 2])

    bounds=np.array((
        (0.01, 0.0001, 0, 0, 0, 0, 0, 0),
        (0.06, 0.02, 1, 10, 0.2, 0.1, np.pi/2, 2.3)
    ))
    result = least_squares(residuals, params_initial, args=(data,), ftol=1e-12, xtol=1e-12,
                        # bounds=(
                        #      np.zeros(7),
                        #      np.ones(7)*100),
                        # )
                           bounds=bounds)
    print("Optimized parameters:", result.x)
    print("Optimized params scaled to bounds:", (result.x - bounds[0]) / (bounds[1] - bounds[0]))
    print("Final predicted values:", make_predictions(result.x, data))
    print("Residuals: ", residuals(result.x, data))
    print("Final info:", result)



