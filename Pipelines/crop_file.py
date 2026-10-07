from viconnexusapi import ViconNexus

# ---------------------------------------------------------
# Configuration: Target Marker Set for Cropping Evaluation
# ---------------------------------------------------------
CROPPING_MARKERS = [
    # Thorax
    'C7', 'T10', 'CLAV', 'STRN', 'LSHO', 'RSHO',
    # Pelvis
    'LASI', 'RASI', 'LPSI', 'RPSI', 'RPEL', 'LPEL',
    # Left Lower Limb
    'LTHIA', 'LKNE', 'LTHI',
    'LTIBA', 'LANK', 'LTIB',
    'LHEE', 'LTOE',
    # Right Lower Limb
    'RTHIA', 'RKNE', 'RTHI',
    'RTIBA', 'RANK', 'RTIB',
    'RHEE', 'RTOE'
]

# ---------------------------------------------------------
# Refactored Cropping Function
# ---------------------------------------------------------
def crop_trial_by_marker_completeness(required_markers=None, completion_ratio=1.0, frame_buffer=3):
    """
    Crops the active trial based on marker set presence rather than a raw numeric count.

    :param required_markers: List of marker names that must be present. Defaults to CROPPING_MARKERS.
    :param completion_ratio: Float (0.0 to 1.0). 1.0 requires 100% of required_markers to be present.
    :param frame_buffer: Safety margin (in frames) to pad/trim at start and end boundaries.
    """
    vicon = ViconNexus.ViconNexus()
    subjects = vicon.GetSubjectNames()
    if not subjects:
        print("Error: No subject found in the active trial.")
        return

    subject = subjects[0]
    eval_markers = required_markers if required_markers else CROPPING_MARKERS

    # Load trajectory existence states for target markers
    marker_exists = {}
    total_frames = None
    missing_from_model = []

    for m in eval_markers:
        try:
            _, _, _, e = vicon.GetTrajectory(subject, m)
            marker_exists[m] = list(e)
            if total_frames is None:
                total_frames = len(e)
        except Exception:
            missing_from_model.append(m)

    if missing_from_model:
        print(f"Warning: {len(missing_from_model)} marker(s) not found in subject model: {missing_from_model}")

    if not marker_exists or total_frames is None:
        print("Error: None of the target cropping markers were found in the active trial.")
        return

    # Calculate target required count based on available evaluated markers
    available_markers = list(marker_exists.keys())
    target_required_count = int(np.ceil(len(available_markers) * completion_ratio))

    print(f"Evaluating Cropping for Subject '{subject}':")
    print(f"  > Target Markers Present in Model: {len(available_markers)}/{len(eval_markers)}")
    print(
        f"  > Completion Criterion: {target_required_count}/{len(available_markers)} markers present ({completion_ratio * 100:.0f}%)")

    # Evaluate frame-by-frame completeness
    valid_frames = []
    for f in range(total_frames):
        # Check if all (or required ratio) of the target markers exist at frame f
        present_count = sum(1 for m in available_markers if marker_exists[m][f])

        if present_count >= target_required_count:
            valid_frames.append(f)

    if not valid_frames:
        print("Warning: Trial never met the full marker completeness criteria. Cropping aborted.")
        return

    # Determine start and end boundaries with safety buffer
    raw_start = valid_frames[0]
    raw_end = valid_frames[-1]

    crop_start = min(raw_start + frame_buffer, raw_end)
    crop_end = max(raw_end - frame_buffer, crop_start)

    print(f"Auto-Crop Range Detected:")
    print(f"  > Raw Valid Window: Frame {raw_start} to {raw_end}")
    print(
        f"  > Final Cropped Range (with {frame_buffer}-frame buffer): Frame {crop_start} to {crop_end} (Total: {crop_end - crop_start + 1} frames)")

    # Apply Region of Interest to Nexus
    try:
        vicon.SetTrialRegionOfInterest(crop_start, crop_end)
        print("Trial successfully cropped in Nexus.")
    except Exception as e:
        print(f"Error applying trial range in Nexus: {e}")


# ---------------------------------------------------------
# Execution Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    # Option 1: Require 100% of defined markers to be present simultaneously
    crop_trial_by_marker_completeness(completion_ratio=1.0, frame_buffer=3)

    # Option 2: Require 95% of defined markers to be present (allows 1-2 minor dropouts at start/end)
    # crop_trial_by_marker_completeness(completion_ratio=0.95, frame_buffer=3)