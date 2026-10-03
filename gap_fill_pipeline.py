import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from viconnexusapi import ViconNexus

# ---------------------------------------------------------
# 1. Configuration (Easily add new segments here)
# ---------------------------------------------------------
SEGMENTS = {
    'Thorax': {
        'primary': ['C7', 'T10', 'CLAV', 'STRN'],
        'emergency': ['LSHO', 'RSHO'],
        'hierarchy': {
            'C7': ['T10', 'CLAV', 'STRN'],
            'T10': ['C7', 'STRN', 'CLAV'],
            'CLAV': ['STRN', 'C7', 'T10'],
            'STRN': ['CLAV', 'T10', 'C7']
        },
        # 30mm per frame at 100Hz = 3 m/s. Adjust if using higher framerates.
        'max_displacement_mm': 30.0
    },
    'Pelvis': {
        'primary': ['LASI', 'RASI', 'LPSI', 'RPSI'],
        'emergency': ['SACR'],  # If using a sacral tracking marker
        'hierarchy': {
            'LASI': ['RASI', 'LPSI', 'RPSI'],
            'RASI': ['LASI', 'RPSI', 'LPSI'],
            'LPSI': ['RPSI', 'LASI', 'RASI'],
            'RPSI': ['LPSI', 'RASI', 'LASI']
        },
        'max_displacement_mm': 35.0
    }
}


# ---------------------------------------------------------
# 2. Math & Logic Helpers
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
    length = len(segment)
    if length < 5:
        return segment

    window_length = min(length if length % 2 != 0 else length - 1, 15)
    polyorder = 3 if window_length > 3 else 2

    # Pad edges to prevent polynomial "whip" at boundaries
    pad_size = 5
    padded = np.concatenate(([segment[0]] * pad_size, segment, [segment[-1]] * pad_size))
    smoothed = savgol_filter(padded, window_length, polyorder, mode='interp')

    return smoothed[pad_size: -pad_size]


def get_weighted_rigid_transform(A, B, weights):
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

    return R, centroid_B.T - np.dot(R, centroid_A.T)


def apply_dual_pattern_fill(target_data, donor_data, gap):
    start, end = gap['start'], gap['end']

    # Dynamic anchor search: Find closest frames where BOTH markers exist
    anc_pre = next((f for f in range(start - 1, -1, -1) if target_data['e'][f] and donor_data['e'][f]), None)
    anc_post = next((f for f in range(end + 1, len(target_data['e'])) if target_data['e'][f] and donor_data['e'][f]),
                    None)

    has_pre = anc_pre is not None
    has_post = anc_post is not None

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
# 3. Core Processing Engine
# ---------------------------------------------------------
def clean_cluster(vicon, subject, cluster_name, config):
    print(f"--- Processing Cluster: {cluster_name} ---")
    primary_markers = config['primary']
    emergency_markers = config['emergency']
    donor_preferences = config['hierarchy']
    max_displacement = config['max_displacement_mm']

    all_markers = primary_markers + emergency_markers

    # Load trajectory data for this cluster
    track_data = {}
    for m in all_markers:
        try:
            x, y, z, e = vicon.GetTrajectory(subject, m)
            track_data[m] = {'x': list(x), 'y': list(y), 'z': list(z), 'e': list(e)}
        except Exception:
            print(f"Warning: Marker {m} not found in trial. Skipping.")
            return

    total_frames = len(track_data[primary_markers[0]]['e'])

    all_gaps = []
    for m in primary_markers:
        for g in get_gaps(track_data[m]['e']):
            all_gaps.append({'marker': m, 'start': g['start'], 'end': g['end'], 'length': g['length']})

    while all_gaps:
        all_gaps.sort(key=lambda g: g['length'])
        progress_made = False
        unfilled_gaps = []

        # 1. Blended Distance-Weighted Rigid Body Fill
        for gap in all_gaps:
            if progress_made:
                unfilled_gaps.append(gap)
                continue

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

                pose_pre = {
                    m: np.array([track_data[m]['x'][f_pre], track_data[m]['y'][f_pre], track_data[m]['z'][f_pre]]) for m
                    in valid_donors + [m_target]}
                pose_post = {
                    m: np.array([track_data[m]['x'][f_post], track_data[m]['y'][f_post], track_data[m]['z'][f_post]])
                    for m in valid_donors + [m_target]}

                A_pts_pre = np.array([pose_pre[d] for d in valid_donors])
                weights_pre = 1.0 / (np.linalg.norm(A_pts_pre - pose_pre[m_target], axis=1) + 1e-6)

                A_pts_post = np.array([pose_post[d] for d in valid_donors])
                weights_post = 1.0 / (np.linalg.norm(A_pts_post - pose_post[m_target], axis=1) + 1e-6)

                recon_x, recon_y, recon_z = [], [], []
                spike_detected = False

                for i in range(gap['start'], gap['end'] + 1):
                    B_pts = np.array(
                        [[track_data[d]['x'][i], track_data[d]['y'][i], track_data[d]['z'][i]] for d in valid_donors])

                    R_pre, t_pre = get_weighted_rigid_transform(A_pts_pre, B_pts, weights_pre)
                    pos_pre = np.dot(R_pre, pose_pre[m_target].T) + t_pre.T

                    R_post, t_post = get_weighted_rigid_transform(A_pts_post, B_pts, weights_post)
                    pos_post = np.dot(R_post, pose_post[m_target].T) + t_post.T

                    weight = (i - gap['start'] + 1) / (gap['end'] - gap['start'] + 2)
                    merged = (1.0 - weight) * pos_pre + weight * pos_post

                    # Biomechanical Spike Rejection Check
                    if i == gap['start'] and i > 0 and track_data[m_target]['e'][i - 1]:
                        prev_pos = np.array([track_data[m_target]['x'][i - 1],
                                             track_data[m_target]['y'][i - 1],
                                             track_data[m_target]['z'][i - 1]])
                        if np.linalg.norm(merged - prev_pos) > max_displacement:
                            spike_detected = True
                            break

                    recon_x.append(merged[0]);
                    recon_y.append(merged[1]);
                    recon_z.append(merged[2])

                if spike_detected:
                    unfilled_gaps.append(gap)
                    continue

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

        # 2. Dual-Anchor Pattern Fill Fallback
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
            print(f"  > Warning: {len(all_gaps)} {cluster_name} gaps could not be dynamically filled.")
            break

    # 3. Spline Cleanup & Push to Nexus
    print(f"  > Running final Spline pass for {cluster_name}...")
    for m in primary_markers:
        for gap in get_gaps(track_data[m]['e']):
            if gap['length'] <= 5:
                apply_spline_fill(track_data[m], gap)

    print(f"  > Pushing {cluster_name} back to Nexus...")
    for m in primary_markers:
        vicon.SetTrajectory(subject, m,
                            [float(v) for v in track_data[m]['x']],
                            [float(v) for v in track_data[m]['y']],
                            [float(v) for v in track_data[m]['z']],
                            [bool(v) for v in track_data[m]['e']])


# ---------------------------------------------------------
# 4. Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    try:
        vicon = ViconNexus.ViconNexus()
        subject = vicon.GetSubjectNames()[0]

        print(f"Starting Data Cleaning Pipeline for Subject: {subject}")

        # Sequentially clean all defined segments
        for segment_name, config in SEGMENTS.items():
            clean_cluster(vicon, subject, segment_name, config)

        print("Pipeline Execution Complete.")
    except Exception as e:
        print(f"Pipeline Error: {e}")