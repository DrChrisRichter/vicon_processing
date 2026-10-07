import numpy as np
import ezc3d
import os

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
            'C7':   ['T10', 'CLAV', 'STRN'],
            'T10':  ['C7', 'STRN', 'CLAV'],
            'CLAV': ['STRN', 'C7', 'T10'],
            'STRN': ['CLAV', 'T10', 'C7']
        },
        'max_displacement_mm': 30.0
    },
    'Pelvis': {
        'primary': ['LASI', 'RASI', 'LPSI', 'RPSI', 'RPEL', 'LPEL'],
        'emergency': ['RPEL', 'LPEL'],
        'hierarchy': {
            'LASI': ['RASI', 'LPSI', 'RPSI'],
            'RASI': ['LASI', 'RPSI', 'LPSI'],
            'LPSI': ['RPSI', 'LASI', 'RASI'],
            'RPSI': ['LPSI', 'RASI', 'LASI'],
            'RPEL': ['RPSI', 'LASI', 'RASI'],
            'LPEL': ['LPSI', 'RASI', 'LASI']
        },
        'max_displacement_mm': 35.0
    },
    'Left_Shank': {
        'primary': ['LTIBA', 'LANK', 'LTIB', 'LMM'], # LEMA = Lateral Epicondyle / Knee joint center proxy
        'emergency': ['LKNE'],
        'hierarchy': {
            'LTIBA': ['LANK', 'LTIB', 'LKNE'],
            'LANK':  ['LTIBA', 'LTIB', 'LKNE'],
            'LTIB':  ['LTIBA', 'LANK', 'LKNE'],
            'LMM':   ['LTIBA', 'LANK', 'LKNE', 'LTIB']
        },
        'max_displacement_mm': 45.0
    },
    'Right_Shank': {
        'primary': ['RTIBA', 'RANK', 'RTIB', 'RMM'],
        'emergency': ['RKNE'],
        'hierarchy': {
            'RTIBA': ['RANK', 'RTIB', 'RKNE'],
            'RANK':  ['RTIBA', 'RTIB', 'RKNE'],
            'RTIB':  ['RTIBA', 'RANK', 'RKNE'],
            'RMM':   ['RTIBA', 'RANK', 'RKNE', 'RTIB']
        },
        'max_displacement_mm': 45.0
    },
    'Left_Thigh': {
        'primary': ['LTHIA', 'LKNE', 'LTHI'],
        'emergency': [],
        'hierarchy': {
            'LTHIA': ['LKNE', 'LTHI'],
            'LKNE':  ['LTHIA', 'LTHI'],
            'LTHI':  ['LTHIA', 'LKNE']
        },
        'max_displacement_mm': 40.0 # Thighs undergo higher linear acceleration
    },
    'Right_Thigh': {
        'primary': ['RTHIA', 'RKNE', 'RTHI'],
        'emergency': [],
        'hierarchy': {
            'RTHIA': ['RKNE', 'RTHI'],
            'RKNE':  ['RTHIA', 'RTHI'],
            'RTHI':  ['RTHIA', 'RKNE']
        },
        'max_displacement_mm': 40.0
    },
    'Left_Foot': {
        'primary': ['LHEE', 'LTOE', 'LANK'],
        'emergency': [],
        'hierarchy': {
            'LHEE': ['LTOE', 'LANK'],
            'LTOE': ['LHEE', 'LANK'],
            'LANK': ['LHEE', 'LTOE']
        },
        'max_displacement_mm': 50.0 # High impact velocity during ground contact
    },
    'Right_Foot': {
        'primary': ['RHEE', 'RTOE', 'RANK'],
        'emergency': [],
        'hierarchy': {
            'RHEE': ['RTOE', 'RANK'],
            'RTOE': ['RHEE', 'RANK'],
            'RANK': ['RHEE', 'RTOE']
        },
        'max_displacement_mm': 50.0
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

def rigid_fill_gap(track_data, m_target, gap, valid_donors, pose_pre, pose_post, max_displacement):
    """
    Performs distance-weighted Kabsch rigid body reconstruction for a specified gap.
    Blends pre- and post-gap transformations and applies edge-padded Savitzky-Golay smoothing.
    """
    start_frame, end_frame = gap['start'], gap['end']
    gap_length = end_frame - start_frame + 1

    A_pts_pre = np.array([pose_pre[d] for d in valid_donors])
    weights_pre = 1.0 / (np.linalg.norm(A_pts_pre - pose_pre[m_target], axis=1) + 1e-6)

    A_pts_post = np.array([pose_post[d] for d in valid_donors])
    weights_post = 1.0 / (np.linalg.norm(A_pts_post - pose_post[m_target], axis=1) + 1e-6)

    recon_x, recon_y, recon_z = [], [], []

    for i in range(start_frame, end_frame + 1):
        B_pts = np.array([[track_data[d]['x'][i], track_data[d]['y'][i], track_data[d]['z'][i]] for d in valid_donors])

        R_pre, t_pre = get_weighted_rigid_transform(A_pts_pre, B_pts, weights_pre)
        pos_pre = np.dot(R_pre, pose_pre[m_target].T) + t_pre.T

        R_post, t_post = get_weighted_rigid_transform(A_pts_post, B_pts, weights_post)
        pos_post = np.dot(R_post, pose_post[m_target].T) + t_post.T

        # Blend pre and post poses across gap
        weight = (i - start_frame + 1) / (gap_length + 1)
        merged = (1.0 - weight) * pos_pre + weight * pos_post

        # Biomechanical Spike Rejection
        if i == start_frame and i > 0 and track_data[m_target]['e'][i - 1]:
            prev_pos = np.array([track_data[m_target]['x'][i - 1],
                                 track_data[m_target]['y'][i - 1],
                                 track_data[m_target]['z'][i - 1]])
            if np.linalg.norm(merged - prev_pos) > max_displacement:
                return False, "Spike detected exceeding displacement threshold."

        recon_x.append(float(merged[0]))
        recon_y.append(float(merged[1]))
        recon_z.append(float(merged[2]))

    # Verify output dimensions match gap length before writing
    assert len(recon_x) == gap_length, f"Length mismatch: {len(recon_x)} vs {gap_length}"

    # Write reconstructed positions back to gap range
    for idx, i in enumerate(range(start_frame, end_frame + 1)):
        track_data[m_target]['x'][i] = recon_x[idx]
        track_data[m_target]['y'][i] = recon_y[idx]
        track_data[m_target]['z'][i] = recon_z[idx]
        track_data[m_target]['e'][i] = True

    return True, "Success"

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
            recon_x.append(x_pre)
            recon_y.append(y_pre)
            recon_z.append(z_pre)
        else:
            recon_x.append(x_post)
            recon_y.append(y_post)
            recon_z.append(z_post)

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
        data['x'][i], data['y'][i], data['z'][i] = float(x_spline(i)), float(y_spline(i)), float(z_spline(i))
        data['e'][i] = True
    return True

# ---------------------------------------------------------
# 3. Static Reference Reconstruction Helpers
# ---------------------------------------------------------

def read_c3d(path):
    """
    Reads a C3D file using ezc3d and returns a structured object
    containing marker_data mapped to marker names.
    """
    c3d_obj = ezc3d.c3d(path)

    # Extract 3D point data: shape (4, n_markers, n_frames) -> [X, Y, Z, Residual]
    points = c3d_obj['data']['points']
    labels = c3d_obj['parameters']['POINT']['LABELS']['value']

    marker_data = {}
    for idx, label in enumerate(labels):
        # Extract X, Y, Z components across all frames -> shape (n_frames, 3)
        trajectory = points[:3, idx, :].T
        marker_data[label] = trajectory

    class C3DContainer:
        def __init__(self, data_dict):
            self.marker_data = data_dict

    return C3DContainer(marker_data)

def extract_static_marker_reference(vicon_api, file_path=None):
    """
    Locates the static calibration C3D file in the active trial directory,
    extracts pristine 3D coordinates for all available markers, and returns
    a reference dictionary mapping marker names to their baseline 3D positions.
    """
    if file_path is None:
        full_trial_path, _ = vicon_api.GetTrialName()
        if not full_trial_path:
            raise RuntimeError("No active trial loaded in Vicon Nexus.")
        file_path = os.path.dirname(full_trial_path)

    static_files = sorted([
        f for f in os.listdir(file_path)
        if "static" in f.lower() and f.lower().endswith(".c3d") and not f.startswith("._")
    ])

    if not static_files:
        raise FileNotFoundError(f"No static reference C3D file found in {file_path}")

    static_c3d_path = os.path.join(file_path, static_files[0])
    print(f"Extracting static reference markers from: {static_files[0]}")

    static_data = read_c3d(path=static_c3d_path)
    static_reference = {}

    for marker_name, trajectory in static_data.marker_data.items():
        if len(trajectory) == 0:
            continue

        valid_frames = [pos for pos in trajectory if not np.all(pos == 0)]
        if len(valid_frames) > 0:
            static_reference[marker_name] = np.mean(valid_frames, axis=0)

    print(f"Extracted {len(static_reference)} reference markers from static trial.")
    return static_reference

def reconstruct_missing_marker_from_static(vicon, subject_name, target_marker, segment_config, static_ref_poses):
    """
    Reconstructs a completely missing marker (0% active frames) in a dynamic trial
    by calculating its rigid geometry from active donor markers using static reference positions.
    """
    print(f"  > Reconstructing completely missing marker '{target_marker}' from static reference...")

    # Identify segment configuration
    segment_cfg = None
    for seg_name, cfg in segment_config.items():
        if target_marker in cfg['primary'] or target_marker in cfg['emergency']:
            segment_cfg = cfg
            break

    if segment_cfg is None or target_marker not in static_ref_poses:
        return False

    candidate_donors = segment_cfg['primary'] + segment_cfg['emergency']
    valid_static_donors = [d for d in candidate_donors if d != target_marker and d in static_ref_poses]

    # Pull dynamic trajectories for donors
    dynamic_donors = {}
    total_frames = None
    for d in valid_static_donors:
        try:
            x, y, z, e = vicon.GetTrajectory(subject_name, d)
            if any(e):
                dynamic_donors[d] = {'x': list(x), 'y': list(y), 'z': list(z), 'e': list(e)}
                if total_frames is None:
                    total_frames = len(e)
        except Exception:
            continue

    if total_frames is None or len(dynamic_donors) < 3:
        print(f"  > Warning: Insufficient dynamic donor markers available to reconstruct '{target_marker}'.")
        return False

    target_static_pos = static_ref_poses[target_marker]
    active_donors_list = list(dynamic_donors.keys())
    A_pts_static = np.array([static_ref_poses[d] for d in active_donors_list])

    distances = np.linalg.norm(A_pts_static - target_static_pos, axis=1)
    weights = 1.0 / (distances + 1e-6)

    recon_x, recon_y, recon_z, recon_e = [], [], [], []

    for f in range(total_frames):
        active_donors = [d for d in active_donors_list if dynamic_donors[d]['e'][f]]

        if len(active_donors) >= 3:
            active_idx = [active_donors_list.index(d) for d in active_donors]
            A_pts = A_pts_static[active_idx]
            w_pts = weights[active_idx]

            B_pts = np.array([[dynamic_donors[d]['x'][f],
                               dynamic_donors[d]['y'][f],
                               dynamic_donors[d]['z'][f]] for d in active_donors])

            R, t = get_weighted_rigid_transform(A_pts, B_pts, w_pts)
            pos_reconstructed = np.dot(R, target_static_pos.T) + t.T

            recon_x.append(float(pos_reconstructed[0]))
            recon_y.append(float(pos_reconstructed[1]))
            recon_z.append(float(pos_reconstructed[2]))
            recon_e.append(True)
        else:
            recon_x.append(0.0)
            recon_y.append(0.0)
            recon_z.append(0.0)
            recon_e.append(False)

    # Create slot if needed and push back to Nexus
    try:
        vicon.CreateModeledMarker(subject_name, target_marker)
    except Exception:
        pass

    vicon.SetTrajectory(
        subject_name, target_marker,
        [float(v) for v in recon_x],
        [float(v) for v in recon_y],
        [float(v) for v in recon_z],
        [bool(v) for v in recon_e]
    )
    print(f"  > Successfully reconstructed '{target_marker}' using static geometry.")
    return True

# ---------------------------------------------------------
# 4. Core Processing Engine
# ---------------------------------------------------------

def clean_cluster(vicon, subject, cluster_name, config, static_ref_poses):
    print(f"--- Processing Cluster: {cluster_name} ---")
    primary_markers = config['primary']
    emergency_markers = config['emergency']
    donor_preferences = config['hierarchy']
    max_displacement = config['max_displacement_mm']

    all_markers = primary_markers + emergency_markers

    # Step 1: Reconstruct completely missing markers using static reference pose
    for m in primary_markers:
        try:
            x, y, z, e = vicon.GetTrajectory(subject, m)
            if not any(e):  # Marker missing entirely in dynamic trial
                reconstruct_missing_marker_from_static(vicon, subject, m, SEGMENTS, static_ref_poses)
        except Exception:
            # Marker trajectory does not exist in Nexus trial
            reconstruct_missing_marker_from_static(vicon, subject, m, SEGMENTS, static_ref_poses)

    # Load updated trajectory data for cluster
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

                success, _ = rigid_fill_gap(track_data, m_target, gap, valid_donors, pose_pre, pose_post, max_displacement)
                if success:
                    progress_made = True
                else:
                    unfilled_gaps.append(gap)
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
                for donor in donor_preferences.get(m_target, []):
                    if donor in track_data and apply_dual_pattern_fill(track_data[m_target], track_data[donor], gap):
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

    # 3. Spline Cleanup Pass
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
# 5. Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    try:
        vicon = ViconNexus.ViconNexus()
        subject = vicon.GetSubjectNames()[0]

        print(f"Starting Data Cleaning Pipeline for Subject: {subject}")

        # Extract static reference once at startup
        static_ref_poses = extract_static_marker_reference(vicon)

        # Sequentially clean all defined segments
        for segment_name, config in SEGMENTS.items():
            clean_cluster(vicon, subject, segment_name, config, static_ref_poses)

        print("Pipeline Execution Complete.")
    except Exception as e:
        print(f"Pipeline Execution Failed: {e}")
        print(f"Pipeline Error: {e}")