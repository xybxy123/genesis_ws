import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Define and check file path ---
file_name = "wheel_leg_imu_data.npz"
# Assume the script is in the same directory as the data file
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, file_name)

print(f"Attempting to load data file: {file_path}")

if not os.path.exists(file_path):
    print(f"❌ Error: Data file not found at: {file_path}")
    exit()

try:
    # --- 2. Load data ---
    data_archive = np.load(file_path, allow_pickle=True)
    print("✅ Data file loaded successfully!")

    # 3. Extract key data
    if 'timestamp' not in data_archive.files:
        print("❌ Error: Missing 'timestamp' key in the file.")
        exit()
        
    timestamps = data_archive['timestamp']
    
    # Check and extract linear acceleration and angular velocity
    lin_acc = data_archive.get('lin_acc')
    ang_vel = data_archive.get('ang_vel')

    if lin_acc is None or ang_vel is None:
        print("❌ Error: Missing 'lin_acc' or 'ang_vel' key in the file.")
        exit()

    # Ensure time axis and data axes have matching lengths
    min_len = min(len(timestamps), len(lin_acc), len(ang_vel))
    timestamps = timestamps[:min_len]
    lin_acc = lin_acc[:min_len]
    ang_vel = ang_vel[:min_len]

    # --- 4. Calculate time intervals (dt) for integration ---
    # Compute time differences between consecutive samples
    dt = np.diff(timestamps)  # Shape: (n_samples - 1,)
    # Ensure dt is positive (handle potential time glitches)
    dt = np.maximum(dt, 1e-6)  # Replace negative/zero values with small epsilon

    # --- 5. Integrate IMU data ---
    # ------------------------------
    # Linear Acceleration → Velocity → Position
    # ------------------------------
    # Integrate acceleration to get velocity (trapezoidal rule)
    # Initial velocity set to 0 (can be modified if initial condition is known)
    velocity = np.zeros_like(lin_acc)
    for i in range(3):  # Integrate for X/Y/Z axes
        # np.cumsum: cumulative sum (trapezoidal integration for discrete data)
        velocity[1:, i] = np.cumsum((lin_acc[:-1, i] + lin_acc[1:, i]) / 2 * dt)

    # Integrate velocity to get position
    # Initial position set to 0 (can be modified if initial condition is known)
    position = np.zeros_like(lin_acc)
    for i in range(3):
        position[1:, i] = np.cumsum((velocity[:-1, i] + velocity[1:, i]) / 2 * dt)

    # ------------------------------
    # Angular Velocity → Orientation (Angle)
    # ------------------------------
    # Integrate angular velocity to get angle (trapezoidal rule)
    # Initial angle set to 0 (can be modified if initial condition is known)
    angle = np.zeros_like(ang_vel)
    for i in range(3):
        angle[1:, i] = np.cumsum((ang_vel[:-1, i] + ang_vel[1:, i]) / 2 * dt)
    # Convert angles to degrees for better readability (optional)
    angle_deg = np.rad2deg(angle)

    # --- 6. Create plots (original + integrated data) ---
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    
    # Define colors and labels
    colors = ['r', 'g', 'b']
    axes_labels = ['X-axis', 'Y-axis', 'Z-axis']

    # ------------------------------
    # Plot 1: Linear Acceleration (Original)
    # ------------------------------
    ax_acc = axes[0]
    ax_acc.set_title(f'Linear Acceleration (IMU: {lin_acc.shape[0]} samples)', fontweight='bold')
    ax_acc.set_ylabel('Acceleration ($m/s^2$)')
    for i in range(3):
        ax_acc.plot(timestamps, lin_acc[:, i], label=axes_labels[i], color=colors[i], linewidth=1.2)
    ax_acc.legend(loc='upper right')
    ax_acc.grid(True, linestyle='--', alpha=0.6)

    # ------------------------------
    # Plot 2: Velocity (Integrated from Acceleration)
    # ------------------------------
    ax_vel = axes[1]
    ax_vel.set_title('Velocity (Integrated from Acceleration)', fontweight='bold')
    ax_vel.set_ylabel('Velocity ($m/s$)')
    for i in range(3):
        ax_vel.plot(timestamps, velocity[:, i], label=axes_labels[i], color=colors[i], linewidth=1.2)
    ax_vel.legend(loc='upper right')
    ax_vel.grid(True, linestyle='--', alpha=0.6)

    # ------------------------------
    # Plot 3: Position (Integrated from Velocity)
    # ------------------------------
    ax_pos = axes[2]
    ax_pos.set_title('Position (Integrated from Velocity)', fontweight='bold')
    ax_pos.set_ylabel('Position ($m$)')
    for i in range(3):
        ax_pos.plot(timestamps, position[:, i], label=axes_labels[i], color=colors[i], linewidth=1.2)
    ax_pos.legend(loc='upper right')
    ax_pos.grid(True, linestyle='--', alpha=0.6)

    # ------------------------------
    # Plot 4: Angular Position (Integrated from Angular Velocity)
    # ------------------------------
    ax_angle = axes[3]
    ax_angle.set_title('Angular Position (Integrated from Angular Velocity)', fontweight='bold')
    ax_angle.set_xlabel('Time (seconds)')
    ax_angle.set_ylabel('Angular Position (degrees)')  # Using degrees for readability
    for i in range(3):
        ax_angle.plot(timestamps, angle_deg[:, i], label=axes_labels[i], color=colors[i], linewidth=1.2)
    ax_angle.legend(loc='upper right')
    ax_angle.grid(True, linestyle='--', alpha=0.6)

    # Add drift warning text (important for IMU integration)
    fig.text(0.5, 0.01, 
             '⚠️ Note: IMU integration suffers from drift. For better results, use sensor fusion (e.g., Kalman filter) with GPS/IMU fusion.',
             ha='center', fontsize=10, style='italic', color='darkred')

    # Adjust layout and display
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])  # Leave space for warning text
    plt.show()

    # Optional: Save integrated data to NPZ file
    save_integrated = input("\nDo you want to save integrated data (velocity/position/angle)? (y/n): ").strip().lower()
    if save_integrated == 'y':
        output_file = "integrated_imu_data.npz"
        np.savez(output_file,
                 timestamp=timestamps,
                 lin_acc=lin_acc,
                 ang_vel=ang_vel,
                 velocity=velocity,
                 position=position,
                 angle_rad=angle,
                 angle_deg=angle_deg)
        print(f"✅ Integrated data saved to: {output_file}")

except Exception as e:
    print(f"❌ Error occurred while processing data: {e}")
    import traceback
    traceback.print_exc()  # Print detailed error stack for debugging
