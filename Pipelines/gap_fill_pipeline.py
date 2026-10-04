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
    'Left_Thigh': {
        'primary': ['LTHIA', 'LKNE', 'MKNE'],
        'emergency': ['LASI', 'LPSI'],
        'hierarchy': {
            'LTHIA': ['LKNE', 'MKNE', 'LASI'],
            'LKNE':  ['LTHIA', 'MKNE', 'LASI'],
            'MKNE':  ['LTHIA', 'LKNE', 'LASI']
        },
        'max_displacement_mm': 40.0 # Thighs undergo higher linear acceleration
    },
    'Right_Thigh': {
        'primary': ['RTHIA', 'RKNE', 'RMKN'],
        'emergency': ['RASI', 'RPSI'],
        'hierarchy': {
            'RTHIA': ['RKNE', 'RMKN', 'RASI'],
            'RKNE':  ['RTHIA', 'RMKN', 'RASI'],
            'RMKN':  ['RTHIA', 'RKNE', 'RASI']
        },
        'max_displacement_mm': 40.0
    },
    'Left_Shank': {
        'primary': ['LTIBA', 'LANK', 'LEMA'], # LEMA = Lateral Epicondyle / Knee joint center proxy
        'emergency': ['LKNE', 'MKNE'],
        'hierarchy': {
            'LTIBA': ['LANK', 'LEMA', 'LKNE'],
            'LANK':  ['LTIBA', 'LEMA', 'LKNE'],
            'LEMA':  ['LTIBA', 'LANK', 'LKNE']
        },
        'max_displacement_mm': 45.0
    },
    'Right_Shank': {
        'primary': ['RTIBA', 'RANK', 'REMA'],
        'emergency': ['RKNE', 'RMKN'],
        'hierarchy': {
            'RTIBA': ['RANK', 'REMA', 'RKNE'],
            'RANK':  ['RTIBA', 'REMA', 'RKNE'],
            'REMA':  ['RTIBA', 'RANK', 'RKNE']
        },
        'max_displacement_mm': 45.0
    },
    'Left_Foot': {
        'primary': ['LHEE', 'LTOE', 'LANK'],
        'emergency': ['LMETA'], # 5th Metatarsal if available
        'hierarchy': {
            'LHEE': ['LTOE', 'LANK'],
            'LTOE': ['LHEE', 'LANK'],
            'LANK': ['LHEE', 'LTOE']
        },
        'max_displacement_mm': 50.0 # High impact velocity during ground contact
    },
    'Right_Foot': {
        'primary': ['RHEE', 'RTOE', 'RANK'],
        'emergency': ['RMETA'],
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
# 4. Static Reference Reconstructor
# ---------------------------------------------------------

def estimate_missing_marker(api, subject_name, file_path, file_name):
    """
    Extracts reference marker geometry from a static calibration C3D file and uses
    it to reconstruct markers that were removed or missing in subsequent dynamic trials.
    """
    # 1. Read the dynamic C3D file
    dynamic_path = os.path.join(file_path, file_name)
    data = read_c3d(path=dynamic_path)

    # Identify markers defined in the model but completely missing from this dynamic trial
    missing_markers = [
        x for x in data.biomechanical_mdl.get_mdl_marker()
        if x not in data.marker_data.keys() or len(data.marker_data[x]) == 0
    ]

    if len(missing_markers) == 0:
        print("\tNo missing removed-markers detected.")
        return

    # 2. Locate and read the static reference C3D file
    static_files = [x for x in os.listdir(file_path) if "static" in x.lower() and x.lower().endswith(".c3d")]
    assert len(static_files) != 0, "No Static Reference Found"
    assert len(static_files) == 1, f"Multiple Static References Found: {','.join(static_files)}"

    static_file = static_files[0]
    static_ref = read_c3d(path=os.path.join(file_path, static_file))

    if static_ref.biomechanical_mdl.name != data.biomechanical_mdl.name:
        raise LookupError('Static and Dynamic files do not share the same marker set model')

    print(f"\tExtracting static reference pose from {static_file}...")

    # Extract clean baseline positions from the static calibration trial
    static_pose = {}
    for m in static_ref.marker_data.keys():
        valid_frames = [pos for pos in static_ref.marker_data[m] if not np.all(pos == 0)]
        if valid_frames:
            # Use mean position across valid static frames as the baseline geometry
            static_pose[m] = np.mean(valid_frames, axis=0)

    # Verify total frame range of active dynamic trial
    total_frames = api.GetTrialRange()[1] - api.GetTrialRange()[0] + 1

    # 3. Segment fill removed markers into dynamic trial
    for target_marker in missing_markers:
        if target_marker not in static_pose:
            print(f"\t  > Marker {target_marker} missing in static reference. Cannot reconstruct.")
            continue

        # Find segment configuration containing the removed target marker
        segment_cfg = None
        target_segment = None
        for seg_name, cfg in SEGMENTS.items():
            if target_marker in cfg['primary'] or target_marker in cfg['emergency']:
                target_segment = seg_name
                segment_cfg = cfg
                break

        if segment_cfg is None:
            print(f"\t  > Marker {target_marker} not assigned to any segment in SEGMENTS dict. Skipping.")
            continue

        # Find donor markers present in both static ref and dynamic trial
        candidate_donors = segment_cfg['primary'] + segment_cfg['emergency']
        donors = [d for d in candidate_donors if d != target_marker and d in static_pose]

        # Load donor trajectory streams from dynamic trial
        dynamic_donors = {}
        for d in donors:
            try:
                x, y, z, e = api.GetTrajectory(subject_name, d)
                dynamic_donors[d] = {'x': list(x), 'y': list(y), 'z': list(z), 'e': list(e)}
            except Exception:
                continue

        # Calculate static reference distances and inverse weights relative to target marker
        A_pts_static = np.array([static_pose[d] for d in donors if d in dynamic_donors])
        target_static_pos = static_pose[target_marker]

        if len(A_pts_static) < 3:
            print(f"\t  > Insufficient active donors in dynamic trial to reconstruct {target_marker} (need >= 3).")
            continue

        valid_donors = [d for d in donors if d in dynamic_donors]
        distances = np.linalg.norm(A_pts_static - target_static_pos, axis=1)
        weights = 1.0 / (distances + 1e-6)

        # Reconstruct trajectory frame by frame
        recon_x, recon_y, recon_z, recon_e = [], [], [], []

        for f in range(total_frames):
            active_donors = [d for d in valid_donors if dynamic_donors[d]['e'][f]]

            if len(active_donors) >= 3:
                active_idx = [valid_donors.index(d) for d in active_donors]
                A_pts = A_pts_static[active_idx]
                w_pts = weights[active_idx]

                B_pts = np.array([[dynamic_donors[d]['x'][f],
                                   dynamic_donors[d]['y'][f],
                                   dynamic_donors[d]['z'][f]] for d in active_donors])

                # Calculate distance-weighted Kabsch transform using static baseline reference
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

        # Smooth output trajectory segment
        recon_x = smooth_gap_segment(recon_x)
        recon_y = smooth_gap_segment(recon_y)
        recon_z = smooth_gap_segment(recon_z)

        # Create slot and push trajectory back to Nexus
        try:
            api.CreateModeledMarker(subject_name, target_marker)
        except Exception:
            pass

        vicon_format = [recon_x, recon_y, recon_z]
        api.SetModelOutput(subject_name, target_marker, vicon_format, recon_e)
        print(f"  > Reconstructed removed marker '{target_marker}' using static geometry ({target_segment} segment).")

    print("\tStatic reference segment fill complete.")


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