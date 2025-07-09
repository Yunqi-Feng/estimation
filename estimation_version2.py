import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.signal import find_peaks

APADP = 10 * np.log10(PADP_refined)

PDP_lin = np.sum(PADP_refined, axis=0)
PAP_lin = np.sum(PADP_refined, axis=1)
PDP_log = 10 * np.log10(PDP_lin)
PAP_log = 10 * np.log10(PAP_lin)
# PDP_log = pdp_log
# PAP_log = 10*np.log10(np.sum(10**(APADP_modified[:, :] / 10), axis=1))
array = PDP_log

# Criterion: Find elements greater than a certain threshold which depends on the superposition of noise floor from each snapshot
criterion = array > -100
# #criterion = array > -100


# # Extract elements that satisfy the criterion
# elements = array[criterion]

# Get the indices of these elements
indices = np.where(criterion)[0]

def group_measurements(delays,angles, powers, gap_threshold):
    """
    Group measured angles (and the corresponding powers) into clusters where the angles
    are contiguous (i.e. differences <= gap_threshold). A gap larger than gap_threshold 
    is assumed to indicate a new arrival.
    
    Parameters:
      angles       : 1D array of measured angles (degrees).
      powers       : 1D array of measured powers (must have the same length as angles).
      gap_threshold: maximum difference (in degrees) between successive angles to be 
                     considered part of the same group.
                     
    Returns:
      groups: a list of tuples, each tuple is (group_angles, group_powers)
    """
    angles = np.array(angles)
    powers = np.array(powers)
    if len(angles) == 0:
        return []
    
    groups = []
    current_angles = [angles[0]]
    current_powers = [powers[0]]
    current_delays = [delays]
    
    for i in range(1, len(angles)):
        if np.abs(angles[i] - angles[i-1]) <= gap_threshold:
            current_angles.append(angles[i])
            current_powers.append(powers[i])
            #current_delays.append(delays)
        else:
            groups.append((np.array(current_angles), np.array(current_powers),np.array(current_delays)))
            current_angles = [angles[i]]
            current_powers = [powers[i]]
            current_delays = [delays]
    
    groups.append((np.array(current_angles), np.array(current_powers),np.array(current_delays)))
    return groups




delays_tmp = indices
powers_angles =[]
angles_meas = []
angles_indices = []
group = []
delays = []
gap_threshold = 1 # depends on stepsize
for i in range(len(delays_tmp)):
    APADP_ang = APADP[:,indices[i]]
    criterion = APADP_ang > -100
    #criterion = APADP_ang > -95
    #criterion = np.logical_and(APADP_ang > (np.max(APADP_ang) - 30), APADP_ang > -95)
    power_angles_tmp = APADP_ang[criterion]
    #print(power_angles_tmp)
    angles_indices_tmp = np.where(criterion)[0]
    angles_meas_tmp = RxAngle[np.where(criterion)[0]]
    # print(angles_meas_tmp)
    group = group_measurements(delays_tmp[i],angles_indices_tmp, power_angles_tmp, 1)
    #print(len(group))
    for i in range(len(group)):
    #powers_angles.append(APADP_ang[criterion])
    #angles_indices.append(np.where(criterion)[0])
        angles_meas.append(RxAngle[group[i][0]])
        powers_angles.append(group[i][1])
        delays.append(group[i][2])
    

def angle_to_index(angle, angle_step=0.1,
                   min_angle=-90.0, max_angle=90.0):
    """
    Convert 'angle' in degrees to an index in an antenna_pattern array
    that goes from -90° to +90° in 0.1° steps (length 1801).
    
    - We clamp the angle to the valid range [-90, 90].
    - Then index = round((angle - (-90)) / 0.1).
    """
    # Clamp angle to the valid range
    angle_clamped = max(min_angle, min(angle, max_angle))
    
    # Convert angle to zero-based index
    # e.g. angle_clamped = -90 => index = 0
    #      angle_clamped = -89.9 => index = 1
    #      angle_clamped =  0    => index = 900
    #      angle_clamped = +90   => index = 1800
    idx = int(round((angle_clamped - min_angle) / angle_step))
    return idx

def deembed_antenna_pattern(
    measured_dirs, 
    measured_power, 
    antenna_pattern,
    start_align, 
    end_align, 
    angle_step=0.1
):
    """
    De-embed the antenna pattern from 4 measured directions/powers,
    using an antenna pattern defined from -90° to +90° in 0.1° steps (length=1801).
    
    We allow measured_dirs to be outside [-90, +90], because we can rotate
    the antenna physically to bring those path directions into the forward beam.
    
    :param measured_dirs: array of size 4 with the measured directions [deg]
    :param measured_power: array of size 4 with the measured powers
    :param antenna_pattern: 1D numpy array of length 1801; 
                           antenna_pattern[i] = gain at angle = -90 + i*0.1 deg
    :param start_align: start angle of alignment, e.g., -12 deg
    :param end_align: end angle of alignment, e.g., 36 deg
    :param angle_step: step size for alignment angle (0.1 deg)
    :return: (best_alignment_angle, min_variance, path_power)
             best_alignment_angle: angle that yields the smallest variance 
             min_variance: the value of that smallest variance
             path_power: the average of the 4 weighted powers
                         (one for each measured direction)
                         at the best alignment angle.
    """
    # Number of alignment steps (inclusive of endpoints)
    num_steps = int(round((end_align - start_align) / angle_step)) + 1
    
    # Prepare a matrix: 4 rows (for 4 measured directions),
    #                   num_steps columns (one per alignment angle)
    weighted_matrix = np.zeros((len(measured_dirs), num_steps))
    
    # Generate all alignment angles from start_align to end_align
    alignment_angles = [
        start_align + i * angle_step for i in range(num_steps)
    ]
    
    # Loop over each alignment angle
    for col_idx, alpha in enumerate(alignment_angles):
        for row_idx, d in enumerate(measured_dirs):
            # offset = alpha - d (in degrees)
            offset_angle = alpha - d
            
            # Convert offset_angle to index in antenna_pattern
            gain_idx = angle_to_index(offset_angle,
                                      angle_step=angle_step,
                                      min_angle=-90.0,
                                      max_angle=90.0)
            # Retrieve antenna gain
            gain = antenna_pattern[gain_idx]
            
            # Weighted power = measured_power * gain
            weighted_matrix[row_idx, col_idx] = measured_power[row_idx] / gain
    
    # Compute variance of each column (i.e., variance of 4 weighted values)
    variances = np.var(weighted_matrix, axis=0)
    
    # Find column index of the smallest variance
    min_var_idx = np.argmin(variances)
    min_var = variances[min_var_idx]
    
    # This is the best alignment angle
    best_alignment_angle = alignment_angles[min_var_idx]
    
    # Path power at the best alignment angle:
    # average of the four weighted powers
    path_power = 10 * np.log10(np.mean(weighted_matrix[:, min_var_idx]))-3.2
    
    return best_alignment_angle, path_power

# -------------------------------------------------------------------
# Example usage (with made-up data):
if __name__ == "__main__":
    Angle_estimation = np.zeros(len(angles_meas))
    Path_gain = np.zeros(len(angles_meas))
    for i in range(len(angles_meas)):
    # Example measured directions
        measured_dirs = angles_meas[i]
        print(measured_dirs)
        print(delays[i])
        # Example measured powers (linear or dB—depends on your usage)
        measured_power = 10 ** (powers_angles[i]/10)
        # Create a mock antenna_pattern from -90 to +90 in 0.1° steps (length=1801).
        # Let's define a simple main lobe at 0° for demonstration.
        # You should replace this with your real measured or simulated pattern.
        angles = np.asarray(theta)

        # Example: max gain = 1 at 0°, then it decreases towards ±90°
        # This is just a toy example. Real patterns would be different.
        antenna_pattern = np.asarray(gain)
        # Normalized so that antenna_pattern[900] (i.e. 0 deg) = 1
        #breakpoint()
        #%debug
        start_align = np.min(measured_dirs)
        end_align = np.max(measured_dirs)
        # De-embed the pattern
        best_angle, path_power = deembed_antenna_pattern(
            measured_dirs, 
            measured_power, 
            antenna_pattern, 
            start_align, 
            end_align, 
            angle_step=0.1
        )
        Delay_estimation = np.concatenate(delays)/20
        Angle_estimation[i] = best_angle
        Path_gain[i] = path_power
        print(f"Best alignment angle: {best_angle:.2f} deg")
        print(f"True path power: {path_power:.2f} dB")
        #weighted_matrix = 10 * np.log10(weighted_matrix)
        #print(weighted_matrix)

plt.figure()
plt.scatter(Delay_estimation, Angle_estimation, c = Path_gain, cmap='viridis', s=50, edgecolors='k')
plt.colorbar(label='Path gain [dBm]')
plt.xlabel("Delay [ns]")
plt.xlim(0, 100)
plt.ylim(-180, 180)
plt.ylabel("Angle [degree]")
#plt.title("Detected Dominant Paths in Angle-Delay Domain")
plt.grid(True)
plt.show()

# Create a DataFrame
data = pd.DataFrame({
    'Delay': Delay_estimation,
    'Angle': Angle_estimation,
    'Path gain': Path_gain
})
#data.to_csv('estimation_threshold_20dB_below\\data_center_without_clustering.csv', index=False)