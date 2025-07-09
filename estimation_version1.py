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
    circular continuity. The angles are assumed to be in degrees (range -180 to 180)
    but are treated on the 360° circle.

    Parameters:
        delay       : scalar delay value for this measurement (e.g., a delay bin index).
        angles      : 1D array of measured angles (degrees).
        powers      : 1D array of measured powers (must have the same length as angles).
        gap_threshold: maximum allowed circular difference (in degrees) between successive angles 
                      to be considered part of the same group.
    
    Returns:
        groups: a list of tuples (group_angles, group_powers, delay)
                where group_angles and group_powers are np.array objects.
                
    Notes:
      - The function first sorts the angles (and associated powers) by angle.
      - It uses circular_difference() so that—for example—angles near –180 and 180 can be
        grouped if their circular difference is less than gap_threshold.
    """
    if len(angles) == 0:
        return []
    
    # Sort the measurements by angle:
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
    
    # Now adjust any group that spans the -180/180 boundary.  
    # For any group in which the difference between max and min (in circular sense) is large,
    # we can shift the negative angles upward by 360 (or vice versa) for a unified representation.
    adjusted_groups = []
    for angles_grp, powers_grp, d in groups:
        if len(angles_grp) > 1:
            # Find the raw range:
            raw_diff = angles_grp.max() - angles_grp.min()
            # Compute the "wrapped" range:
            wrapped_diff = 360 - raw_diff if raw_diff > 180 else raw_diff
            # If the raw range is very large but the wrapped range is small,
            # then the group spans the -180/180 discontinuity.
            if raw_diff > wrapped_diff:
                new_angles = np.array([a + 360 if a < 0 else a for a in angles_grp])
                adjusted_groups.append((new_angles, powers_grp, d))
            else:
                adjusted_groups.append((angles_grp, powers_grp, d))
        else:
            adjusted_groups.append((angles_grp, powers_grp, d))
    return adjusted_groups

# -------------------------------
# Merge groups function
# -------------------------------
def merge_groups(groups, region, peak_delay, gap_threshold):
    """
    Merge groups (each a tuple of (angles, powers, delay)) that fall within a specified delay region,
    and then re-run grouping on the merged measurements to ensure circular continuity.
    
    Parameters:
       groups       : list of tuples (angles, powers, delay) from different delay bins.
       region       : tuple (start, end) defining the delay region for merging (inclusive).
       peak_delay   : the peak delay value to assign to the merged group.
       gap_threshold: gap threshold (in degrees) used for grouping.
       
    Returns:
       tuple: (merged_angles, merged_powers, peak_delay)
              where merged_angles and merged_powers are arrays of the merged measurements.
              
    Note:
       If the merged group (from all groups in the region) is not continuous (using the same
       gap threshold), then group_measurements() will split them and we select the sub-group with
       the highest total power.
    """
    merged_angles = np.array([])
    merged_powers = np.array([])

    # Merge all groups whose delay falls within the region.
    for group in groups:
        angles, powers, d = group
        # Here we treat d as a scalar. Make sure the region boundaries are defined appropriately.
        if region[0] <= d <= region[2]:
            merged_angles = np.concatenate([merged_angles, angles])
            merged_powers = np.concatenate([merged_powers, powers])
    
    if len(merged_angles) == 0:
        return np.array([]), np.array([]), peak_delay
    
    # Re-group the merged measurements using our function.
    re_groups = group_measurements(peak_delay, merged_angles, merged_powers, gap_threshold)
    
    # If more than one subgroup is obtained, select the one with highest total power.
    if len(re_groups) > 1:
        group_powers = [np.sum(g[1]) for g in re_groups]
        idx = np.argmax(group_powers)
        merged_angles, merged_powers, _ = re_groups[idx]
    else:
        merged_angles, merged_powers, _ = re_groups[0]
        
    return merged_angles, merged_powers, peak_delay

# -------------------------------
# (Rest of your antenna deembedding algorithm remains)
# -------------------------------
def angle_to_index(angle, angle_step=0.1, min_angle=-90.0, max_angle=90.0):
    """
    Convert an angle (in degrees) to an index for an antenna pattern array that spans
    from min_angle to max_angle (in steps of angle_step). The function operates on numerical
    values and uses a simple linear conversion after clamping the value.

    This function no longer uses complex exponentials.
    
    For example:
      - angle = -90  => index = 0
      - angle =   0  => index = round((0 - (-90))/0.1) = 900
      - angle = +90  => index = 1800
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
    De-embed the antenna pattern from measured directions/powers using an antenna pattern defined 
    for a 360° concept (though the pattern is stored from -90° to +90°) by comparing measured 
    angles with candidate alignment angles. Here measured_dirs (and measured_power) can contain 
    multiple (grouped) directions. The function finds the alignment angle that minimizes variance 
    of weighted power.

    Parameters:
      measured_dirs  : array-like, measured directions in degrees.
      measured_power : array-like, corresponding power values.
      antenna_pattern: 1D numpy array of length 1801 (for angles -90° to +90° in 0.1° steps).
      start_align    : starting candidate alignment angle (degrees).
      end_align      : ending candidate alignment angle (degrees).
      angle_step     : step size for candidate alignment angle (default 0.1°).
    
    Returns:
      (best_alignment_angle, path_power): best alignment angle (degrees) that minimizes variance and 
                                          the corresponding path power (in dB, after weighting).
    """
    # Remove duplicate directions if present (using circular equivalence):
    unique_dirs = []
    unique_powers = []
    processed_angles = set()
    for i, angle in enumerate(measured_dirs):
        # Normalize to [0,360)
        norm_angle = angle % 360
        # Check if we have already processed an equivalent angle (e.g., 0 and 360)
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
    
    # Compute number of candidate alignment angles.
    num_steps = int(round((end_align - start_align) / angle_step)) + 1
    weighted_matrix = np.zeros((len(measured_dirs), num_steps))
    alignment_angles = np.array([start_align + i * angle_step for i in range(num_steps)])
    
    # Loop over candidate alignment angles.
    for col_idx, alpha in enumerate(alignment_angles):
        for row_idx, d in enumerate(measured_dirs):
            offset_angle = alpha - d  # difference in degrees.
            gain_idx = angle_to_index(offset_angle, angle_step=angle_step, min_angle=-90.0, max_angle=90.0)
            gain = antenna_pattern[gain_idx]
            # Here we weight by measured power divided by the gain.
            weighted_matrix[row_idx, col_idx] = measured_power[row_idx] / gain
            
    # Compute column variances.
    variances = np.var(weighted_matrix, axis=0)
    min_var_idx = np.argmin(variances)
    best_alignment_angle = alignment_angles[min_var_idx]
    # Compute path power as a power-weighted average (converted to dB and offset by an empirical constant).
    path_power = 10 * np.log10(np.mean(weighted_matrix[:, min_var_idx])) - 3.2
    
    return best_alignment_angle, path_power

# -------------------------------
# Example usage (your main loop)
# -------------------------------
#if __name__ == "__main__":
    # Assume variables: PADP_refined, RxAngle, theta, gain, regions, detected_peaks_peaks are defined.
    # PADP_refined: a 2D array of power delay profiles.
    # RxAngle: an array representing the mapping from indices to physical angle values.
    # theta: an array of angles for the antenna pattern.
    # gain: an array representing the antenna pattern (length should be 1801).
    # regions: a list of delay regions as tuples (start, end) for merging.
    # detected_peaks_peaks: a list/array of peak delay indices detected in the PDP.
    
    # Convert PADP_refined to dB
APADP = 10 * np.log10(PADP_refined)
PDP_log = 10 * np.log10(np.sum(PADP_refined, axis=0))

# Find indices where PDP_log exceeds a threshold.
criterion = PDP_log > -100
indices = np.where(criterion)[0]

grouped_measurements = defaultdict(list)

# For each detected delay index, get the corresponding angles from APADP.
for i in range(len(indices)):
    delay_val = indices[i]
    APADP_ang = APADP[:, indices[i]]
    #print(measured_dirs)
    #print(delays[i])
    #print()
    # Use a threshold relative to the max in that delay bin.
    criterion_ang = np.logical_and(APADP_ang > (np.max(APADP_ang) - 20), APADP_ang > -95)
    power_angles_tmp = APADP_ang[criterion_ang]
    angles_indices_tmp = np.where(criterion_ang)[0]
    angles_meas_tmp = RxAngle[angles_indices_tmp]
    # Group the angles using our circular grouping function.
    groups = group_measurements(delay_val, angles_indices_tmp, power_angles_tmp, gap_threshold=1)
    for g in groups:
        grouped_measurements[delay_val].append(g)

# Merge groups based on detected regions.
merged_groups = []
angles_meas = []
powers_angles = []
delays = []

for region in regions:
    groups_in_region = []
    for delay in range(region[0], region[2] + 1):
        if delay in grouped_measurements:
            groups_in_region.extend(grouped_measurements[delay])
    #print(groups_in_region)
    # Find the peak delay in the region.
    peak_delay_in_region = None
    for peak in detected_peaks_peaks:
        if region[0] <= peak <= region[2]:
            peak_delay_in_region = peak
            break
    if peak_delay_in_region is not None:
        merged_angle, merged_power, merged_delay = merge_groups(groups_in_region, region, peak_delay_in_region, gap_threshold=1)
        if len(merged_angle) > 0:
            # Convert angle indices (merged_angle) to physical angles using RxAngle.
            angles_meas.append(RxAngle[merged_angle.astype(int)])
            powers_angles.append(merged_power)
            delays.append(merged_delay)
    else:
        print(f"No peak found in region {region}")

# Call the de-embedding routine for each merged group:
Angle_estimation = np.zeros(len(angles_meas))
Path_gain = np.zeros(len(angles_meas))
Delay_estimation = np.zeros(len(angles_meas))

for i in range(len(angles_meas)):
    measured_dirs = angles_meas[i]
    measured_power = 10 ** (powers_angles[i] / 10)  # convert dB to linear if needed
    # Use the provided antenna pattern (array 'gain') and its corresponding angle array 'theta'
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
    Delay_estimation[i] = delays[i] / 20  # Adjust scaling as needed

# Plot the results.
import matplotlib.pyplot as plt
plt.figure()
scatter = plt.scatter(Delay_estimation, Angle_estimation, c=Path_gain, cmap='viridis', s=50, edgecolors='k')
plt.colorbar(scatter, label='Path gain [dBm]')
plt.xlabel("Delay [ns]")
plt.xlim(0, 100)
plt.ylim(-180, 180)
plt.ylabel("Angle [degree]")
plt.grid(True)
plt.show()

# Create a DataFrame (if using pandas)
import pandas as pd
data = pd.DataFrame({
    'Delay': Delay_estimation,
    'Angle': Angle_estimation,
    'Path gain': Path_gain
})