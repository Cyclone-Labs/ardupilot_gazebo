import numpy as np
import math

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
    
    return {
        'total_force': total_force,
        'total_torque': total_torque,
    }


# Example usage
if __name__ == "__main__":
    # Example parameters for a 4-blade propeller
    result = calculate_propeller_forces(
        num_blades=4,                   # 4-blade propeller
        blade_radius=0.5,               # m (50cm from hub to blade CP)
        air_density=1.225,              # kg/m³ (sea level)
        area=0.02,                      # m² (blade area, 200 cm²)
        a0=math.radians(5),             # rad (5 degree built-in pitch angle)
        cla=2*math.pi,                  # 1/rad (theoretical thin airfoil)
        cda=0.01,                       # 1/rad (small drag slope)
        alpha_stall=math.radians(15),   # rad (15 degree stall angle)
        cla_stall=0.0,                  # 1/rad (no lift after stall)
        cda_stall=1.0,                  # 1/rad (high drag after stall)
        angular_velocity=100,           # rad/s (about 955 RPM)
    )
    
    print("Propeller Forces:")
    print(f"Total Force: {result['total_force']} N")
    print(f"Total Torque: {result['total_torque']} N⋅m")
    thrust_z = result['total_force'][2]  # Thrust is typically in Z direction
    torque_z = result['total_torque'][2]  # Torque about Z axis
    print(f"\nThrust (Z): {thrust_z:.2f} N")
    print(f"Torque (Z): {torque_z:.3f} N⋅m")