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

    # 1. Attempt Blended Rigid Body Fill for as many as possible
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
            # Find the closest valid frame BEFORE the gap where all 4 markers exist
            f_pre = None
            for f in range(gap['start'] - 1, -1, -1):
                if all(track_data[m]['e'][f] for m in markers):
                    f_pre = f
                    break

            # Find the closest valid frame AFTER the gap where all 4 markers exist
            f_post = None
            for f in range(gap['end'] + 1, total_frames):
                if all(track_data[m]['e'][f] for m in markers):
                    f_post = f
                    break

            # Handle edge cases (if gap is at the very start or very end of the trial)
            if f_pre is None: f_pre = f_post
            if f_post is None: f_post = f_pre
            if f_pre is None and f_post is None: f_pre = f_post = ref_frame  # Fallback

            # Extract the actual marker relationships (postures) at those specific frames
            pose_pre = {m: np.array([track_data[m]['x'][f_pre], track_data[m]['y'][f_pre], track_data[m]['z'][f_pre]])
                        for m in markers}
            pose_post = {
                m: np.array([track_data[m]['x'][f_post], track_data[m]['y'][f_post], track_data[m]['z'][f_post]]) for m
                in markers}

            # Reconstruct missing target frame by frame using the blended approach
            for i in range(gap['start'], gap['end'] + 1):
                # Current positions of the donor markers
                B_pts = np.array(
                    [[track_data[d]['x'][i], track_data[d]['y'][i], track_data[d]['z'][i]] for d in m_donors])

                # Estimate 1: Using pre-gap posture
                A_pts_pre = np.array([pose_pre[d] for d in m_donors])
                R_pre, t_pre = get_rigid_transform(A_pts_pre, B_pts)
                pos_pre = np.dot(R_pre, pose_pre[m_target].T) + t_pre.T

                # Estimate 2: Using post-gap posture
                A_pts_post = np.array([pose_post[d] for d in m_donors])
                R_post, t_post = get_rigid_transform(A_pts_post, B_pts)
                pos_post = np.dot(R_post, pose_post[m_target].T) + t_post.T

                # Calculate blending weight (0.0 near start, 1.0 near end)
                weight = (i - gap['start'] + 1) / (gap['end'] - gap['start'] + 2)

                # Linearly interpolate between the two estimates
                target_reconstructed = (1.0 - weight) * pos_pre + weight * pos_post

                track_data[m_target]['x'][i] = target_reconstructed[0]
                track_data[m_target]['y'][i] = target_reconstructed[1]
                track_data[m_target]['z'][i] = target_reconstructed[2]
                track_data[m_target]['e'][i] = True

            progress_made = True
            break  # Break and re-evaluate gaps since data changed

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
    # Cast NumPy types back to standard Python floats and booleans
    x_clean = [float(val) for val in track_data[m]['x']]
    y_clean = [float(val) for val in track_data[m]['y']]
    z_clean = [float(val) for val in track_data[m]['z']]
    e_clean = [bool(val) for val in track_data[m]['e']]

    vicon.SetTrajectory(subject, m,
                        x_clean,
                        y_clean,
                        z_clean,
                        e_clean)

print("Pipeline complete.")