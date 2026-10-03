import numpy as np
from scipy.interpolate import CubicSpline
from viconnexusapi import ViconNexus


# ---------------------------------------------------------
# 1. Helper Math & Fill Functions
# ---------------------------------------------------------
def get_gaps(exists_array):
    """Finds all gaps in a boolean existence array."""
    gaps, start, in_gap = [], 0, False
    for i, val in enumerate(exists_array):
        if not val and not in_gap:
            start, in_gap = i, True
        elif val and in_gap:
            gaps.append({'start': start, 'end': i - 1, 'length': i - start})
            in_gap = False
    if in_gap:
        gaps.append({'start': start, 'end': len(exists_array) - 1, 'length': len(exists_array) - start})
    return gaps


def apply_spline_fill(data, gap, pad=5):
    """Fills a gap using a cubic spline based on surrounding frames."""
    start, end = gap['start'], gap['end']
    pre_start = max(0, start - pad)
    post_end = min(len(data['e']) - 1, end + pad)

    # Gather valid indices around the gap
    valid_idx = [i for i in range(pre_start, post_end + 1) if data['e'][i] and (i < start or i > end)]
    if len(valid_idx) < 4:
        return False  # Not enough points for a cubic spline

    x_spline = CubicSpline(valid_idx, [data['x'][i] for i in valid_idx])
    y_spline = CubicSpline(valid_idx, [data['y'][i] for i in valid_idx])
    z_spline = CubicSpline(valid_idx, [data['z'][i] for i in valid_idx])

    for i in range(start, end + 1):
        data['x'][i], data['y'][i], data['z'][i] = x_spline(i), y_spline(i), z_spline(i)
        data['e'][i] = True
    return True


def get_rigid_transform(A, B):
    """Calculates Rotation and Translation to move point cloud A to B using Kabsch."""
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    H = np.dot((A - centroid_A).T, (B - centroid_B))
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = centroid_B.T - np.dot(R, centroid_A.T)
    return R, t


def apply_pattern_fill(target_data, donor_data, gap):
    """Fills a gap by copying the displacement pattern of a donor marker."""
    start, end = gap['start'], gap['end']
    anchor = start - 1  # Frame right before gap

    if anchor < 0 or not target_data['e'][anchor]:
        return False  # Needs a valid anchor frame

    for i in range(start, end + 1):
        if not donor_data['e'][i]: return False  # Donor must exist during gap

        target_data['x'][i] = target_data['x'][anchor] + (donor_data['x'][i] - donor_data['x'][anchor])
        target_data['y'][i] = target_data['y'][anchor] + (donor_data['y'][i] - donor_data['y'][anchor])
        target_data['z'][i] = target_data['z'][anchor] + (donor_data['z'][i] - donor_data['z'][anchor])
        target_data['e'][i] = True
    return True


# ---------------------------------------------------------
# 2. Main Script Logic
# ---------------------------------------------------------
vicon = ViconNexus.ViconNexus()
subject = vicon.GetSubjectNames()[0]
markers = ['C7', 'T10', 'CLAV', 'STRN']

# Load trajectory data into a dictionary
track_data = {}
for m in markers:
    x, y, z, e = vicon.GetTrajectory(subject, m)
    track_data[m] = {'x': list(x), 'y': list(y), 'z': list(z), 'e': list(e)}

total_frames = len(track_data[markers[0]]['e'])

# ---------------------------------------------------------
# STEP A: Spline fill all gaps <= 5 frames
# ---------------------------------------------------------
print("Executing Spline Fills for gaps <= 5 frames...")
for m in markers:
    gaps = get_gaps(track_data[m]['e'])
    for gap in gaps:
        if gap['length'] <= 5:
            print(f'fill {m}: {gap}')
            apply_spline_fill(track_data[m], gap)

# ---------------------------------------------------------
# STEP B: Extract Reference Pose for Rigid Body Fills
# ---------------------------------------------------------
# Find a "perfect" frame where all 4 markers exist to serve as the rigid body reference
ref_frame = None
for i in range(total_frames):
    if all(track_data[m]['e'][i] for m in markers):
        ref_frame = i
        break

if ref_frame is None:
    raise ValueError("No frame found where all 4 thorax markers exist simultaneously.")

# ---------------------------------------------------------
# STEP C: Iterative Rigid Body and Pattern Fill Loop
# ---------------------------------------------------------
print("Executing Iterative Rigid Body and Pattern Fills...")
while True:
    # Gather all remaining gaps across the 4 markers
    all_gaps = []
    for m in markers:
        for g in get_gaps(track_data[m]['e']):
            all_gaps.append({'marker': m, 'start': g['start'], 'end': g['end'], 'length': g['length']})

    if not all_gaps:
        print("All Thorax gaps successfully filled.")
        break

    # Rank remaining gaps by length (shortest first)
    all_gaps.sort(key=lambda g: g['length'])
    progress_made = False

    # 1. Attempt Rigid Body Fill for as many as possible
    for gap in all_gaps:
        m_target = gap['marker']
        m_donors = [m for m in markers if m != m_target]

        # Check if the 3 donor markers exist for the entire duration of the gap
        can_rigid_fill = True
        for i in range(gap['start'], gap['end'] + 1):
            if not all(track_data[d]['e'][i] for d in m_donors):
                can_rigid_fill = False
                break

        if can_rigid_fill:
            # Reconstruct missing target frame by frame
            print('fill')
            for i in range(gap['start'], gap['end'] + 1):
                A_pts = np.array([ref_pose[d] for d in m_donors])
                B_pts = np.array(
                    [[track_data[d]['x'][i], track_data[d]['y'][i], track_data[d]['z'][i]] for d in m_donors])

                R, t = get_rigid_transform(A_pts, B_pts)

                target_reconstructed = np.dot(R, ref_pose[m_target].T) + t.T
                track_data[m_target]['x'][i] = target_reconstructed[0]
                track_data[m_target]['y'][i] = target_reconstructed[1]
                track_data[m_target]['z'][i] = target_reconstructed[2]
                track_data[m_target]['e'][i] = True

            progress_made = True
            break  # Break and re-evaluate gaps since data changed

    if progress_made:
        continue  # Restart while loop

    # 2. If no Rigid Body fill is possible, find the smallest gap to Pattern Fill
    # This might unlock a Rigid Body fill in the next iteration.
    for gap in all_gaps:

        m_target = gap['marker']
        m_donors = [m for m in markers if m != m_target]

        pattern_success = False
        for donor in m_donors:
            if apply_pattern_fill(track_data[m_target], track_data[donor], gap):
                pattern_success = True
                break

        if pattern_success:
            progress_made = True
            break

    if not progress_made:
        print("Warning: Stopped iterating. Remaining gaps cannot be filled with current methods.")
        break

# ---------------------------------------------------------
# STEP D: Write Data Back to Nexus
# ---------------------------------------------------------
print("Writing data back to Nexus...")
for m in markers:
    vicon.SetTrajectory(subject, m,
                        track_data[m]['x'],
                        track_data[m]['y'],
                        track_data[m]['z'],
                        track_data[m]['e'])

print("Pipeline complete.")