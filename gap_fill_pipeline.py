import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from viconnexusapi import ViconNexus


# ---------------------------------------------------------
# 1. Helper Functions
# ---------------------------------------------------------
def get_gaps(exists_array):
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


def smooth_gap_segment(segment):
    """Uses a Savitzky-Golay filter to smooth fills while preserving peak impacts."""
    length = len(segment)
    if length < 5:
        return segment  # Too short to filter without distortion

    # Window length must be odd and less than or equal to segment length
    window_length = min(length if length % 2 != 0 else length - 1, 15)
    polyorder = 3 if window_length > 3 else 2

    return savgol_filter(segment, window_length, polyorder, mode='interp')


def get_weighted_rigid_transform(A, B, weights):
    """Calculates Rotation and Translation using a distance-weighted Kabsch algorithm."""
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    W = weights.reshape(-1, 1)

    centroid_A = np.sum(A * W, axis=0)
    centroid_B = np.sum(B * W, axis=0)

    A_centered = A - centroid_A
    B_centered = B - centroid_B

    H = np.dot(A_centered.T, (B_centered * W))
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    t = centroid_B.T - np.dot(R, centroid_A.T)
    return R, t


def apply_dual_pattern_fill(target_data, donor_data, gap):
    start, end = gap['start'], gap['end']
    anc_pre, anc_post = start - 1, end + 1

    has_pre = anc_pre >= 0 and target_data['e'][anc_pre] and donor_data['e'][anc_pre]
    has_post = anc_post < len(target_data['e']) and target_data['e'][anc_post] and donor_data['e'][anc_post]

    for i in range(start, end + 1):
        if not donor_data['e'][i]: return False

    if not has_pre and not has_post: return False

    recon_x, recon_y, recon_z = [], [], []
    for i in range(start, end + 1):
        if has_pre:
            x_pre = target_data['x'][anc_pre] + (donor_data['x'][i] - donor_data['x'][anc_pre])
            y_pre = target_data['y'][anc_pre] + (donor_data['y'][i] - donor_data['y'][anc_pre])
            z_pre = target_data['z'][anc_pre] + (donor_data['z'][i] - donor_data['z'][anc_pre])
        if has_post:
            x_post = target_data['x'][anc_post] + (donor_data['x'][i] - donor_data['x'][anc_post])
            y_post = target_data['y'][anc_post] + (donor_data['y'][i] - donor_data['y'][anc_post])
            z_post = target_data['z'][anc_post] + (donor_data['z'][i] - donor_data['z'][anc_post])

        if has_pre and has_post:
            w = (i - start + 1) / (end - start + 2)
            recon_x.append((1.0 - w) * x_pre + w * x_post)
            recon_y.append((1.0 - w) * y_pre + w * y_post)
            recon_z.append((1.0 - w) * z_pre + w * z_post)
        elif has_pre:
            recon_x.append(x_pre);
            recon_y.append(y_pre);
            recon_z.append(z_pre)
        else:
            recon_x.append(x_post);
            recon_y.append(y_post);
            recon_z.append(z_post)

    recon_x = smooth_gap_segment(recon_x)
    recon_y = smooth_gap_segment(recon_y)
    recon_z = smooth_gap_segment(recon_z)

    for idx, i in enumerate(range(start, end + 1)):
        target_data['x'][i], target_data['y'][i], target_data['z'][i] = recon_x[idx], recon_y[idx], recon_z[idx]
        target_data['e'][i] = True
    return True


def apply_spline_fill(data, gap, pad=5):
    start, end = gap['start'], gap['end']
    pre_start, post_end = max(0, start - pad), min(len(data['e']) - 1, end + pad)
    valid_idx = [i for i in range(pre_start, post_end + 1) if data['e'][i] and (i < start or i > end)]

    if len(valid_idx) < 4: return False
    x_spline = CubicSpline(valid_idx, [data['x'][i] for i in valid_idx])
    y_spline = CubicSpline(valid_idx, [data['y'][i] for i in valid_idx])
    z_spline = CubicSpline(valid_idx, [data['z'][i] for i in valid_idx])

    for i in range(start, end + 1):
        data['x'][i], data['y'][i], data['z'][i] = x_spline(i), y_spline(i), z_spline(i)
        data['e'][i] = True
    return True


# ---------------------------------------------------------
# 2. Initialization & Data Loading
# ---------------------------------------------------------
vicon = ViconNexus.ViconNexus()
subject = vicon.GetSubjectNames()[0]

primary_markers = ['C7', 'T10', 'CLAV', 'STRN']
emergency_markers = ['LSHO', 'RSHO']
all_markers = primary_markers + emergency_markers

donor_preferences = {
    'C7': ['T10', 'CLAV', 'STRN'],
    'T10': ['C7', 'STRN', 'CLAV'],
    'CLAV': ['STRN', 'C7', 'T10'],
    'STRN': ['CLAV', 'T10', 'C7']
}

track_data = {}
for m in all_markers:
    x, y, z, e = vicon.GetTrajectory(subject, m)
    track_data[m] = {'x': list(x), 'y': list(y), 'z': list(z), 'e': list(e)}
total_frames = len(track_data[primary_markers[0]]['e'])

vicon.DisplayMessage("Initializing optimized data cleaning pipeline...")

# Initialize gap list once to prevent heavy CPU recalculation cycles
all_gaps = []
for m in primary_markers:
    for g in get_gaps(track_data[m]['e']):
        all_gaps.append({'marker': m, 'start': g['start'], 'end': g['end'], 'length': g['length']})

# ---------------------------------------------------------
# STEP A: Iterative Blended Rigid Body & Dual Pattern Fill
# ---------------------------------------------------------
while all_gaps:
    all_gaps.sort(key=lambda g: g['length'])
    progress_made = False

    # 1. Blended Distance-Weighted Rigid Body Fill
    unfilled_gaps = []
    for gap in all_gaps:
        m_target = gap['marker']
        m_donors_primary = [m for m in primary_markers if m != m_target]

        valid_donors = [d for d in m_donors_primary if
                        all(track_data[d]['e'][i] for i in range(gap['start'], gap['end'] + 1))]
        if len(valid_donors) < 3:
            valid_emergency = [d for d in emergency_markers if
                               all(track_data[d]['e'][i] for i in range(gap['start'], gap['end'] + 1))]
            valid_donors.extend(valid_emergency)

        if len(valid_donors) >= 3:
            f_pre = next((f for f in range(gap['start'] - 1, -1, -1) if
                          all(track_data[m]['e'][f] for m in valid_donors + [m_target])), None)
            f_post = next((f for f in range(gap['end'] + 1, total_frames) if
                           all(track_data[m]['e'][f] for m in valid_donors + [m_target])), None)

            if f_pre is None: f_pre = f_post
            if f_post is None: f_post = f_pre
            if f_pre is None:
                unfilled_gaps.append(gap)
                continue

            pose_pre = {m: np.array([track_data[m]['x'][f_pre], track_data[m]['y'][f_pre], track_data[m]['z'][f_pre]])
                        for m in valid_donors + [m_target]}
            pose_post = {
                m: np.array([track_data[m]['x'][f_post], track_data[m]['y'][f_post], track_data[m]['z'][f_post]]) for m
                in valid_donors + [m_target]}

            A_pts_pre = np.array([pose_pre[d] for d in valid_donors])
            weights_pre = 1.0 / (np.linalg.norm(A_pts_pre - pose_pre[m_target], axis=1) + 1e-6)

            A_pts_post = np.array([pose_post[d] for d in valid_donors])
            weights_post = 1.0 / (np.linalg.norm(A_pts_post - pose_post[m_target], axis=1) + 1e-6)

            recon_x, recon_y, recon_z = [], [], []
            for i in range(gap['start'], gap['end'] + 1):
                B_pts = np.array(
                    [[track_data[d]['x'][i], track_data[d]['y'][i], track_data[d]['z'][i]] for d in valid_donors])

                R_pre, t_pre = get_weighted_rigid_transform(A_pts_pre, B_pts, weights_pre)
                pos_pre = np.dot(R_pre, pose_pre[m_target].T) + t_pre.T

                R_post, t_post = get_weighted_rigid_transform(A_pts_post, B_pts, weights_post)
                pos_post = np.dot(R_post, pose_post[m_target].T) + t_post.T

                weight = (i - gap['start'] + 1) / (gap['end'] - gap['start'] + 2)
                merged = (1.0 - weight) * pos_pre + weight * pos_post
                recon_x.append(merged[0]);
                recon_y.append(merged[1]);
                recon_z.append(merged[2])

            recon_x = smooth_gap_segment(recon_x)
            recon_y = smooth_gap_segment(recon_y)
            recon_z = smooth_gap_segment(recon_z)

            for idx, i in enumerate(range(gap['start'], gap['end'] + 1)):
                track_data[m_target]['x'][i] = recon_x[idx]
                track_data[m_target]['y'][i] = recon_y[idx]
                track_data[m_target]['z'][i] = recon_z[idx]
                track_data[m_target]['e'][i] = True

            progress_made = True
        else:
            unfilled_gaps.append(gap)

    all_gaps = unfilled_gaps
    if progress_made: continue

    # 2. Dual-Anchor Pattern Fill
    unfilled_gaps = []
    for gap in all_gaps:
        if not progress_made:
            m_target = gap['marker']
            pattern_success = False
            for donor in donor_preferences[m_target]:
                if apply_dual_pattern_fill(track_data[m_target], track_data[donor], gap):
                    pattern_success = True
                    break

            if pattern_success:
                progress_made = True
                continue
        unfilled_gaps.append(gap)

    all_gaps = unfilled_gaps
    if not progress_made:
        vicon.DisplayMessage(f"Pipeline stopped: {len(all_gaps)} thorax gaps could not be dynamically filled.")
        break

# ---------------------------------------------------------
# STEP B: Spline Fill Cleanup
# ---------------------------------------------------------
vicon.DisplayMessage("Running final Spline smoothing pass...")
for m in primary_markers:
    # Get remaining small gaps dynamically
    for gap in get_gaps(track_data[m]['e']):
        if gap['length'] <= 5:
            apply_spline_fill(track_data[m], gap)

# ---------------------------------------------------------
# STEP C: Write Primary Markers Back to Nexus
# ---------------------------------------------------------
for m in primary_markers:
    vicon.SetTrajectory(subject, m,
                        [float(v) for v in track_data[m]['x']],
                        [float(v) for v in track_data[m]['y']],
                        [float(v) for v in track_data[m]['z']],
                        [bool(v) for v in track_data[m]['e']])

vicon.DisplayMessage("Thorax cleaning pipeline complete.")