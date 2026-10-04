import numpy as np
from viconnexusapi import ViconNexus


# ---------------------------------------------------------
# Helper Functions: Force Signal & Frame Synchronization
# ---------------------------------------------------------
def get_subject_mass(vicon, subject):
    """Retrieves subject body mass in kg from Nexus parameters."""
    try:
        return float(vicon.GetSubjectParam(subject, "Bodymass")[0])
    except Exception:
        return 70.0  # Default fallback mass in kg


def get_resultant_fz(vicon):
    """Sum vertical ground reaction forces (Fz) across all connected force plates."""
    total_fz = None
    try:
        device_ids = vicon.GetDeviceIDs()
        for dev_id in device_ids:
            name, dev_type, rate, _, _, _ = vicon.GetDeviceDetails(dev_id)
            if "Force" in dev_type or "Plate" in name:
                # Channel 3 is standard vertical Fz in Nexus
                fz, _, _ = vicon.GetDeviceChannel(dev_id, 1, 3)
                fz_array = np.abs(np.array(fz))
                if total_fz is None:
                    total_fz = fz_array
                else:
                    total_fz += fz_array
    except Exception as e:
        print(f"Error extracting force plate data: {e}")

    if total_fz is None:
        raise RuntimeError("No force plate Fz signal detected in trial.")

    return total_fz


def transform_fp_to_mocap_frame(vicon, fp_frame):
    """Converts a force plate sample index to the corresponding Mocap frame index."""
    if fp_frame is None:
        return None

    mocap_rate = vicon.GetFrameRate()
    fp_rate = 1000.0  # Standard fallback

    try:
        for dev_id in vicon.GetDeviceIDs():
            name, dev_type, rate, _, _, _ = vicon.GetDeviceDetails(dev_id)
            if "Force" in dev_type or "Plate" in name:
                fp_rate = rate
                break
    except Exception:
        pass

    ratio = fp_rate / mocap_rate
    return int(round(fp_frame / ratio))


# ---------------------------------------------------------
# Exercise Event Analyzers
# ---------------------------------------------------------
import numpy as np
from scipy.signal import find_peaks
from viconnexusapi import ViconNexus


def register_squat_events(vicon, subject, min_rep_distance_sec=1.5, min_depth_drop_mm=100.0):
    """
    Detects and registers multi-repetition Squat events based on Center of Mass (COM) Z-height:
    - Squat Bottom: Minimum Z-point (peak knee/hip flexion) for each repetition.
    - Squat Start: Frame where COM drops below 98% of standing height prior to bottom.
    - Squat End: Frame where COM returns to 98% of standing height following bottom.
    """
    print("--- Registering Multi-Rep Squat Events ---")
    frame_rate = vicon.GetFrameRate()

    # 1. Extract Center of Mass Z-trajectory (or Pelvis / LASI proxy if COM output not modeled)
    com_z = None
    try:
        # Attempt to pull modeled Center of Mass trajectory
        _, _, z, e = vicon.GetModelOutput(subject, "CenterOfMass")
        com_z = np.array(z)
    except Exception:
        # Fallback to Pelvis marker (LASI) if CenterOfMass is not present
        try:
            _, _, z, e = vicon.GetTrajectory(subject, "LASI")
            com_z = np.array(z)
            print("  > Using LASI marker trajectory as Center of Mass proxy.")
        except Exception as err:
            print(f"Error: Unable to extract COM or Pelvis Z-trajectory: {err}")
            return False

    if com_z is None or len(com_z) == 0:
        print("Error: Empty Z-trajectory array.")
        return False

    # 2. Identify all local minima (bottom of squats)
    # Invert signal so troughs become peaks for scipy.signal.find_peaks
    inverted_com_z = -com_z
    min_samples_between_reps = int(min_rep_distance_sec * frame_rate)

    # Find peaks on inverted signal (minima on original COM signal)
    bottom_frames, _ = find_peaks(
        inverted_com_z,
        distance=min_samples_between_reps,
        prominence=min_depth_drop_mm  # Ensures noise/sway isn't registered as a rep
    )

    if len(bottom_frames) == 0:
        print("Warning: No distinct squat repetitions detected.")
        return False

    print(f"  > Detected {len(bottom_frames)} squat repetition(s).")

    # 3. Process each rep: Calculate 98% standing threshold start and end frames
    mocap_events = []

    for rep_idx, bottom_fp in enumerate(bottom_frames):
        # Determine search bounds for standing baseline
        # Bound backward search to either trial start or previous rep's bottom
        prev_bound = bottom_frames[rep_idx - 1] if rep_idx > 0 else 0
        next_bound = bottom_frames[rep_idx + 1] if rep_idx < len(bottom_frames) - 1 else len(com_z) - 1

        # Peak standing height prior to descending into this rep
        standing_peak_z = np.max(com_z[prev_bound: bottom_fp])
        threshold_98_z = standing_peak_z * 0.98

        # --- Squat Start ---
        # Scan backward from bottom to find frame where COM drops below 98% standing height
        pre_descent = com_z[prev_bound: bottom_fp]
        below_98_idx = np.where(pre_descent >= threshold_98_z)[0]
        start_fp = (prev_bound + below_98_idx[-1]) if len(below_98_idx) > 0 else max(prev_bound,
                                                                                     bottom_fp - int(frame_rate))

        # --- Squat End ---
        # Scan forward from bottom to find frame where COM recovers to 98% standing height
        post_ascent = com_z[bottom_fp: next_bound]
        above_98_idx = np.where(post_ascent >= threshold_98_z)[0]
        end_fp = (bottom_fp + above_98_idx[0]) if len(above_98_idx) > 0 else min(next_bound,
                                                                                 bottom_fp + int(frame_rate))

        mocap_events.append({
            'rep': rep_idx + 1,
            'Start': int(start_fp),
            'Bottom': int(bottom_fp),
            'End': int(end_fp)
        })

    # 4. Write Repetition Events into Nexus
    _write_squat_events_to_nexus(vicon, subject, mocap_events)
    return True


def register_dlcmj_events(vicon, subject, threshold_N=20.0):
    """Detects and registers Double-Leg Countermovement Jump (DLCMJ) events."""
    print("--- Registering DLCMJ Events ---")
    fz_signal = get_resultant_fz(vicon)
    mass = get_subject_mass(vicon, subject)
    bw_N = mass * 9.81

    # 1. Flight Phase (Toe-off & Impact)
    flight_idx = np.where(fz_signal < threshold_N)[0]
    if len(flight_idx) == 0:
        print("Warning: No flight phase detected for DLCMJ.")
        return False

    toeoff_fp = flight_idx[0]
    impact_fp = flight_idx[-1] + 1

    # 2. Movement Start (Unweighting > 5% BW)
    pre_jump = fz_signal[:toeoff_fp]
    unweighting = np.where(np.abs(pre_jump - bw_N) > (bw_N * 0.05))[0]
    start_fp = unweighting[0] if len(unweighting) > 0 else max(0, toeoff_fp - 100)

    # 3. End of Landing (Stabilization post-impact)
    post_impact = fz_signal[impact_fp:]
    end_landing_fp = impact_fp + len(post_impact) - 1
    if len(post_impact) > 50:
        peak_impact = np.argmax(post_impact)
        stabilized = np.where(np.abs(post_impact[peak_impact:] - bw_N) < (bw_N * 0.10))[0]
        if len(stabilized) > 0:
            end_landing_fp = impact_fp + peak_impact + stabilized[0]

    # Convert to Mocap frames
    mocap_events = {
        'Start': transform_fp_to_mocap_frame(vicon, start_fp),
        'Foot Off': transform_fp_to_mocap_frame(vicon, toeoff_fp),
        'Foot Strike': transform_fp_to_mocap_frame(vicon, impact_fp),
        'End Landing': transform_fp_to_mocap_frame(vicon, end_landing_fp)
    }

    # Write events to Nexus
    _write_events_to_nexus(vicon, subject, mocap_events)
    return True


def register_drop_jump_events(vicon, subject, threshold_N=20.0):
    """Detects and registers Drop Jump (DJ) contact and flight phases."""
    print("--- Registering Drop Jump Events ---")
    fz_signal = get_resultant_fz(vicon)

    # Ground contacts (> threshold_N)
    contact_idx = np.where(fz_signal > threshold_N)[0]
    if len(contact_idx) == 0:
        print("Warning: No force plate contact detected for Drop Jump.")
        return False

    initial_contact_fp = contact_idx[0]

    # Second flight phase / takeoff
    flight_after_contact = np.where(fz_signal[initial_contact_fp:] < threshold_N)[0]
    if len(flight_after_contact) == 0:
        takeoff_fp = contact_idx[-1]
        recontact_fp = takeoff_fp
    else:
        takeoff_fp = initial_contact_fp + flight_after_contact[0]
        recontact_idx = np.where(fz_signal[takeoff_fp:] > threshold_N)[0]
        recontact_fp = takeoff_fp + recontact_idx[0] if len(recontact_idx) > 0 else takeoff_fp

    mocap_events = {
        'Initial Contact': transform_fp_to_mocap_frame(vicon, initial_contact_fp),
        'Foot Off': transform_fp_to_mocap_frame(vicon, takeoff_fp),
        'Foot Strike': transform_fp_to_mocap_frame(vicon, recontact_fp)
    }

    _write_events_to_nexus(vicon, subject, mocap_events)
    return True


# ---------------------------------------------------------
# Nexus Writer
# ---------------------------------------------------------
def _write_events_to_nexus(vicon, subject, mocap_events):
    """Clears previous events and creates General + Bilateral events in Nexus."""
    vicon.ClearAllEvents()

    frame_offset = 0.0
    # 1. Write General Context Events
    for event_name, frame in mocap_events.items():
        if frame is not None:
            vicon.CreateAnEvent(subject, 'General', event_name, int(frame), frame_offset)

    print("Successfully written events to Nexus.")


# ---------------------------------------------------------
# Execution Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":

    vicon = ViconNexus.ViconNexus()
    subjects = vicon.GetSubjectNames()
    _, file_name = vicon.GetTrialName()

    if subjects:
        subject = subjects[0]
        # Choose the exercise function to run on the active trial
        if 'CMJ' in file_name.upper():
            register_dlcmj_events(vicon, subject)