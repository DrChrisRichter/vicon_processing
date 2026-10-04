from viconnexusapi import ViconNexus

# ---------------------------------------------------------
# Configuration: All primary markers to evaluate for cropping
# ---------------------------------------------------------
CROPPING_MARKERS = [
    # Thorax
    'C7', 'T10', 'CLAV', 'STRN', 'LSHO', 'RSHO',
    'LASI', 'RASI', 'LPSI', 'RPSI', 'RPEL', 'LPEL',
    'LTHIA', 'LKNE', 'LTHI',
    'RTHIA', 'RKNE', 'RTHI',
    'LTIBA', 'LANK', 'LTIB',
    'RTIBA', 'RANK', 'RTIB',
    'LHEE', 'LTOE',
    'RHEE', 'RTOE'
]


# ---------------------------------------------------------
# Standalone Cropping Function
# ---------------------------------------------------------
def crop_trial(min_required_markers):
    vicon = ViconNexus.ViconNexus()
    subjects = vicon.GetSubjectNames()
    if not subjects:
        print("Error: No subject found in the active trial.")
        return

    subject = subjects[0]
    print(f"Processing Cropping for Subject: {subject} | Min Marker Threshold: {min_required_markers}")

    # Load trajectory existence states for the defined markers
    marker_exists = {}
    total_frames = None

    for m in CROPPING_MARKERS:
        try:
            _, _, _, e = vicon.GetTrajectory(subject, m)
            marker_exists[m] = list(e)
            if total_frames is None:
                total_frames = len(e)
        except Exception:
            # Skip markers not present in the current trial database
            continue

    if not marker_exists or total_frames is None:
        print("Error: Could not load trajectories for cropping evaluation.")
        return

    # Scan frame by frame to find the valid operating window
    valid_start_frame = None
    valid_end_frame = 0

    for f in range(total_frames):
        active_count = sum(1 for m in marker_exists if marker_exists[m][f])

        if active_count >= min_required_markers:
            if valid_start_frame is None:
                valid_start_frame = f  # First frame threshold is met
            valid_end_frame = f  # Updates until the final threshold frame

    # Apply the crop range to Nexus
    if valid_start_frame is not None:
        print(
            f"Auto-Crop Range Detected: Frame {valid_start_frame} to {valid_end_frame} (Total: {valid_end_frame - valid_start_frame + 1} frames)")
        try:
            vicon.SetTrialRegionOfInterest(valid_start_frame, valid_end_frame)
            print("Trial successfully cropped.")
        except Exception as e:
            print(f"Error applying trial range in Nexus: {e}")
    else:
        print("Warning: Trial never reached the minimum active marker count threshold. Cropping aborted.")


# ---------------------------------------------------------
# Execution Entry Point
# ---------------------------------------------------------

if __name__ == "__main__":
    # Supply your minimum active marker count as input here:
    crop_trial(28)