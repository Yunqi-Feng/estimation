import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.signal import find_peaks

# -------------------------------
# Helper for circular (angular) differences
# -------------------------------
def circular_difference(a, b):
    """
    Compute the smallest absolute difference between two angles (in degrees)
    on the circle.
    """
    diff = abs(a - b) % 360
    return diff if diff <= 180 else 360 - diff

# -------------------------------
# Group measurements function (for a single delay bin)
# -------------------------------
def group_measurements(delay, angles, powers, gap_threshold):
    """
    Group measured angles (and their corresponding powers) into clusters based on
    circular continuity. Angles are assumed to be in degrees (range -180 to 180) but
    are treated on the 360° circle.

    Parameters:
      delay        : scalar delay value (e.g., a delay bin index).
      angles       : 1D array of measured angle indices.
      powers       : 1D array of corresponding power values.
      gap_threshold: maximum allowed circular difference (in degrees) between successive angles 
                    to be considered part of the same group.
    
    Returns:
      groups: a list of tuples (group_angles, group_powers, delay)
              where group_angles and group_powers are numpy arrays.
    """
    if len(angles) == 0:
        return []
    
    # Sort the measurements by angle
    sorted_indices = np.argsort(angles)
    sorted_angles = np.array(angles)[sorted_indices]
    sorted_powers = np.array(powers)[sorted_indices]
    
    groups = []
    current_angles = [sorted_angles[0]]
    current_powers = [sorted_powers[0]]
    for i in range(1, len(sorted_angles)):
        gap = circular_difference(sorted_angles[i], sorted_angles[i-1])
        if gap <= gap_threshold:
            current_angles.append(sorted_angles[i])
            current_powers.append(sorted_powers[i])
        else:
            groups.append((np.array(current_angles), np.array(current_powers), delay))
            current_angles = [sorted_angles[i]]
            current_powers = [sorted_powers[i]]
    groups.append((np.array(current_angles), np.array(current_powers), delay))
    
    # Adjust groups that span the -180/180 boundary.
    adjusted_groups = []
    for angles_grp, powers_grp, d in groups:
        if len(angles_grp) > 1:
            raw_diff = angles_grp.max() - angles_grp.min()
            wrapped_diff = 360 - raw_diff if raw_diff > 180 else raw_diff
            if raw_diff > wrapped_diff:
                new_angles = np.array([a + 360 if a < 0 else a for a in angles_grp])
                adjusted_groups.append((new_angles, powers_grp, d))
            else:
                adjusted_groups.append((angles_grp, powers_grp, d))
        else:
            adjusted_groups.append((angles_grp, powers_grp, d))
    return adjusted_groups

# -------------------------------
# NOTE:
# In the previous implementation, merge_groups() combined groups from all delays within a region.
# In the new strategy we discard groups from non-peak delays in each region and preserve only the groups
# from the peak delay. Therefore, we remove or bypass merge_groups() and perform a simple lookup.
# -------------------------------

def angle_to_index(angle, angle_step=0.1, min_angle=-90.0, max_angle=90.0):
    """
    Convert an angle (in degrees) to an index for an antenna pattern
    array that spans from min_angle to max_angle in steps of angle_step.
    
    For example:
      - angle = -90  => index = 0
      - angle = 0    => index = round((0 - (-90))/0.1) = 900
      - angle = 90   => index = 1800
    """
    angle_clamped = max(min_angle, min(angle, max_angle))
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
    De-embed the antenna pattern from measured directions/powers using an antenna
    pattern defined from -90° to 90° in 0.1° steps.
    
    Parameters:
      measured_dirs  : list/array of measured directions (degrees)
      measured_power : list/array of corresponding power values
      antenna_pattern: 1D numpy array of length 1801 for angles -90° to 90°
      start_align    : starting candidate alignment angle (degrees)
      end_align      : ending candidate alignment angle (degrees)
      angle_step     : step size for candidate alignment angle
      
    Returns:
      (best_alignment_angle, path_power):
        best_alignment_angle: alignment angle (degrees) that minimizes variance in the weighted powers
        path_power: resulting path power (dB) after weighting.
    """
    # Remove duplicates using circular equivalence.
    unique_dirs = []
    unique_powers = []
    processed_angles = set()
    for i, angle in enumerate(measured_dirs):
        norm_angle = angle % 360
        if norm_angle not in processed_angles and (norm_angle - 360) not in processed_angles:
            unique_dirs.append(angle)
            unique_powers.append(measured_power[i])
            processed_angles.add(norm_angle)
    
    if len(unique_dirs) < len(measured_dirs):
        measured_dirs = np.array(unique_dirs)
        measured_power = np.array(unique_powers)
    else:
        measured_dirs = np.array(measured_dirs)
        measured_power = np.array(measured_power)
    
    # Generate candidate alignment angles.
    num_steps = int(round((end_align - start_align) / angle_step)) + 1
    weighted_matrix = np.zeros((len(measured_dirs), num_steps))
    alignment_angles = np.array([start_align + i * angle_step for i in range(num_steps)])
    
    for col_idx, alpha in enumerate(alignment_angles):
        for row_idx, d in enumerate(measured_dirs):
            offset_angle = alpha - d
            gain_idx = angle_to_index(offset_angle, angle_step=angle_step, min_angle=-90.0, max_angle=90.0)
            gain = antenna_pattern[gain_idx]
            weighted_matrix[row_idx, col_idx] = measured_power[row_idx] / gain
            
    variances = np.var(weighted_matrix, axis=0)
    min_var_idx = np.argmin(variances)
    best_alignment_angle = alignment_angles[min_var_idx]
    path_power = 10 * np.log10(np.mean(weighted_matrix[:, min_var_idx])) - 3.2
    return best_alignment_angle, path_power

# -------------------------------
# Main processing loop
# -------------------------------
if __name__ == "__main__":
    # Assume that the following variables are defined:
    # PADP_refined: 2D array of refined power delay profiles.
    # RxAngle:  array mapping indices to physical angle values.
    # theta:    array of angles corresponding to the antenna pattern.
    # gain:     array with the antenna pattern (length=1801).
    # regions:  list of delay regions as tuples (start, end) for detection.
    # detected_peaks_peaks: list/array of peak delay indices detected.
    
    # Compute APADP (antenna power delay profile in dB)
    APADP = 10 * np.log10(PADP_refined)
    PDP_log = 10 * np.log10(np.sum(PADP_refined, axis=0))
    
    # Find delay indices above a threshold.
    criterion = PDP_log > -100
    indices = np.where(criterion)[0]
    
    grouped_measurements = defaultdict(list)
    
    # For each delay index, group the measured angle profiles.
    for i in range(len(indices)):
        delay_val = indices[i]
        APADP_ang = APADP[:, delay_val]
        # Threshold on the antenna power relative to the max in that delay bin.
        criterion_ang = np.logical_and(APADP_ang > (np.max(APADP_ang) - 20), APADP_ang > -95)
        power_angles_tmp = APADP_ang[criterion_ang]
        angles_indices_tmp = np.where(criterion_ang)[0]
        angles_meas_tmp = RxAngle[angles_indices_tmp]
        groups = group_measurements(delay_val, angles_indices_tmp, power_angles_tmp, gap_threshold=1)
        for g in groups:
            grouped_measurements[delay_val].append(g)
    
    # -------------------------------
    # NEW STRATEGY:
    # For each detection region, discard groups from all delays except the one corresponding to the peak delay.
    # -------------------------------
    merged_groups = []    # now simply the groups from the peak delay only
    angles_meas = []
    powers_angles = []
    delays = []
    
    for region in regions:
        # region is defined as (start, end); find the peak delay in the region.
        peak_delay_in_region = None
        for peak in detected_peaks_peaks:
            if region[0] <= peak <= region[1]:
                peak_delay_in_region = peak
                break
        if peak_delay_in_region is not None:
            # Retrieve groups only for the peak delay.
            if peak_delay_in_region in grouped_measurements:
                groups_peak = grouped_measurements[peak_delay_in_region]
                # Choose the group with the highest total power.
                group_powers = [np.sum(g[1]) for g in groups_peak]
                idx = np.argmax(group_powers)
                chosen_group = groups_peak[idx]
                merged_angle, merged_power, merged_delay = chosen_group
                if len(merged_angle) > 0:
                    angles_meas.append(RxAngle[merged_angle.astype(int)])
                    powers_angles.append(merged_power)
                    delays.append(merged_delay)
            #else:
                #print(f"No groups for peak delay {peak_delay_in_region}")
        #else:
            #print(f"No peak found in region {region}")
    
    # -------------------------------
    # De-embedding using the peak-delay groups.
    # -------------------------------
    Angle_estimation = np.zeros(len(angles_meas))
    Path_gain = np.zeros(len(angles_meas))
    Delay_estimation = np.zeros(len(angles_meas))
    
    for i in range(len(angles_meas)):
        measured_dirs = angles_meas[i]
        measured_power = 10 ** (powers_angles[i] / 10)  # converting from dB to linear if needed
        antenna_pattern = np.asarray(gain)
        start_align = np.min(measured_dirs)
        end_align = np.max(measured_dirs)
        best_angle, path_power = deembed_antenna_pattern(
            measured_dirs, measured_power, antenna_pattern, start_align, end_align, angle_step=0.1
        )
        print(delays[i])
        print(measured_dirs)
        print(10*np.log10(measured_power))
        print(f"Best alignment angle: {best_angle:.2f} deg")
        print(f"True path power: {path_power:.2f} dB")
        Angle_estimation[i] = best_angle
        Path_gain[i] = path_power
        # Delay scaling: adjust if necessary.
        Delay_estimation[i] = delays[i] / 20
    
    # Plot the results.
    import matplotlib.pyplot as plt
    import pandas as pd
    plt.figure()
    scatter = plt.scatter(Delay_estimation, Angle_estimation, c=Path_gain, cmap='viridis', s=50, edgecolors='k')
    plt.colorbar(scatter, label='Path gain [dBm]')
    plt.xlabel("Delay [ns]")
    plt.xlim(0, 100)
    plt.ylim(-180, 180)
    plt.ylabel("Angle [degree]")
    plt.grid(True)
    plt.show()
    
    data = pd.DataFrame({
        'Delay': Delay_estimation,
        'Angle': Angle_estimation,
        'Path gain': Path_gain
    })