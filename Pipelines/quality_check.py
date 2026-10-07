import os
import numpy as np
import ezc3d
from viconnexusapi import ViconNexus

# ---------------------------------------------------------
# Configuration & Quality Thresholds
# ---------------------------------------------------------
# Define expected segment marker definitions
SEGMENT_MARKERS = {
    'Thorax': ['C7', 'T10', 'CLAV', 'STRN'],
    'Pelvis': ['LASI', 'RASI', 'LPSI', 'RPSI', 'RPEL', 'LPEL'],
    'Left_Shank': ['LTIBA', 'LANK', 'LTIB', 'LMM'],
    'Right_Shank': ['RTIBA', 'RANK', 'RTIB', 'RMM'],
    'Left_Thigh': ['LTHIA', 'LKNE', 'LTHI'],
    'Right_Thigh': ['RTHIA', 'RKNE', 'RTHI'],
    'Left_Foot': ['LHEE', 'LTOE', 'LANK'],
    'Right_Foot': ['RHEE', 'RTOE', 'RANK']
}

# Thresholds for quality flags
MAX_VELOCITY_MM_PER_FRAME = 45.0  # Flags sudden non-physiological jumps
MAX_GEO_DEVIATION_MM = 30.0  # Max distance drift from static baseline geometry


# ---------------------------------------------------------
# Helpers: Static Reference & Audit Checkers
# ---------------------------------------------------------
def read_static_reference(vicon):
    """Extracts mean static positions from the static C3D file in session folder."""
    full_path, _ = vicon.GetTrialName()
    if not full_path:
        raise RuntimeError("No trial currently loaded in Vicon Nexus.")

    session_dir = os.path.dirname(full_path)
    static_files = sorted([
        f for f in os.listdir(session_dir)
        if "static" in f.lower() and f.lower().endswith(".c3d") and not f.startswith("._")
    ])

    if not static_files:
        print("Warning: No static calibration C3D file found for geometry checks.")
        return {}

    c3d_obj = ezc3d.c3d(os.path.join(session_dir, static_files[0]))
    points = c3d_obj['data']['points']
    labels = c3d_obj['parameters']['POINT']['LABELS']['value']

    static_ref = {}
    for idx, label in enumerate(labels):
        traj = points[:3, idx, :].T
        valid = [p for p in traj if not np.all(p == 0)]
        if valid:
            static_ref[label] = np.mean(valid, axis=0)

    return static_ref


def check_marker_quality(m_name, traj, static_ref, segment_markers, max_vel, max_geo_dev):
    """Evaluates gaps, velocity spikes, and inter-marker distance drift."""
    x, y, z, e = traj['x'], traj['y'], traj['z'], traj['e']
    total_frames = len(e)
    issues = []

    # 1. Residual Gap Check
    missing_count = sum(1 for active in e if not active)
    if missing_count > 0:
        pct_missing = (missing_count / total_frames) * 100.0
        issues.append(f"Residual Gaps ({missing_count} frames missing, {pct_missing:.1f}%)")

    # 2. Velocity / Spike Check
    pos = np.column_stack((x, y, z))
    velocities = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    # Mask out velocity transitions across invalid frames
    valid_diffs = np.logical_and(e[:-1], e[1:])
    spikes = np.where((velocities > max_vel) & valid_diffs)[0]

    if len(spikes) > 0:
        max_v = float(np.max(velocities[valid_diffs])) if any(valid_diffs) else 0.0
        issues.append(f"Velocity Spike (Peak: {max_v:.1f} mm/frame across {len(spikes)} boundary transition(s))")

    # 3. Geometric Distance Relationship Check (vs Static Baseline)
    if m_name in static_ref:
        # Find adjacent markers in same segment present in static
        peer_markers = [p for p in segment_markers if p != m_name and p in static_ref]

        geo_drifts = []
        for peer in peer_markers:
            static_dist = np.linalg.norm(static_ref[m_name] - static_ref[peer])

            # Dynamic inter-marker distances frame by frame
            peer_pos = np.column_stack(
                (traj['peer_data'][peer]['x'], traj['peer_data'][peer]['y'], traj['peer_data'][peer]['z']))
            valid_frames = np.logical_and(e, traj['peer_data'][peer]['e'])

            if np.any(valid_frames):
                dyn_dists = np.linalg.norm(pos[valid_frames] - peer_pos[valid_frames], axis=1)
                max_dev = np.max(np.abs(dyn_dists - static_dist))
                geo_drifts.append(max_dev)

        if geo_drifts and max(geo_drifts) > max_geo_dev:
            issues.append(f"Geometric Drift (Max inter-marker displacement: {max(geo_drifts):.1f} mm vs static)")

    return issues


# ---------------------------------------------------------
# Main Audit Execution Script
# ---------------------------------------------------------
def run_quality_check():
    vicon = ViconNexus.ViconNexus()
    subject = vicon.GetSubjectNames()[0]
    trial_path, _ = vicon.GetTrialName()
    trial_name = os.path.basename(trial_path) if trial_path else "Active Trial"

    print(f"\n==================================================")
    print(f"   VICON DATA CLEANING QUALITY AUDIT REPORT       ")
    print(f"   Trial: {trial_name} | Subject: {subject}      ")
    print(f"==================================================\n")

    static_ref = read_static_reference(vicon)
    all_subject_markers = vicon.GetMarkerNames(subject)

    # Cache trajectory data
    track_data = {}
    for m in all_subject_markers:
        try:
            x, y, z, e = vicon.GetTrajectory(subject, m)
            track_data[m] = {'x': np.array(x), 'y': np.array(y), 'z': np.array(z), 'e': np.array(e)}
        except Exception:
            continue

    total_checked = 0
    clean_count = 0
    flagged_summary = []

    for seg_name, markers in SEGMENT_MARKERS.items():
        print(f"--- Segment: {seg_name} ---")

        for m in markers:
            if m not in track_data:
                print(f"  [MISSING] {m:<8} : Marker trajectory not present in trial.")
                flagged_summary.append((m, "Marker Missing entirely"))
                continue

            total_checked += 1

            # Pack peer trajectory context for distance checks
            traj_context = track_data[m].copy()
            traj_context['peer_data'] = {p: track_data[p] for p in markers if p != m and p in track_data}

            # Audit marker
            issues = check_marker_quality(
                m, traj_context, static_ref, markers,
                MAX_VELOCITY_MM_PER_FRAME, MAX_GEO_DEVIATION_MM
            )

            if not issues:
                print(f"  [OK]      {m:<8} : Clean (100% active, smooth, geometrically valid)")
                clean_count += 1
            else:
                issue_str = " | ".join(issues)
                print(f"  [FLAG]    {m:<8} : {issue_str}")
                flagged_summary.append((m, issue_str))

    # Audit Overview Summary
    print(f"\n--------------------------------------------------")
    print(f"AUDIT SUMMARY: {clean_count}/{total_checked} markers passed clean.")
    if flagged_summary:
        print(f"STATUS: WARNING - {len(flagged_summary)} marker(s) flagged for review.")
    else:
        print(f"STATUS: SUCCESS - Trial is fully cleaned and ready for modeling.")
    print(f"--------------------------------------------------\n")


if __name__ == "__main__":
    run_quality_check()