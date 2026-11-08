
import numpy as np
from viconnexusapi import ViconNexus

# ---------------------------------------------------------
# Refactored Cropping Function
# ---------------------------------------------------------
def crop_trial_by_marker_completeness(frame_buffer=3):
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

    (_, _, z), e = vicon.GetModelOutput(subject, "CentreOfMass")

    # Determine start and end boundaries with safety buffer
    raw_start = int(np.where(e)[0][0] + 1)
    raw_end = int(np.where(e)[0][-1] - 1)

    crop_start = min(raw_start, raw_end)
    crop_end = max(raw_end, crop_start)

    # 1. Create a master list of every event name your lab might EVER use.
    try:
        t0 = vicon.GetEvents(subject, 'General', 'Start')[0][0]
        tn = vicon.GetEvents(subject, 'General', 'End')[0][-1]
        frame_rate = vicon.GetFrameRate()
        # 3. Calculate exactly 1 second before the event
        t0 = t0 - frame_rate
        tn = tn + frame_rate

        crop_start = int(max(t0, crop_start))
        crop_end = int(min(tn, crop_end))
    except IndexError:
        pass

    print(f"Auto-Crop Range Detected:")
    print(f"  > Raw Valid Window: Frame {raw_start} to {raw_end}")

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
    crop_trial_by_marker_completeness(frame_buffer=3)

    # Option 2: Require 95% of defined markers to be present (allows 1-2 minor dropouts at start/end)
    # crop_trial_by_marker_completeness(completion_ratio=0.95, frame_buffer=3)