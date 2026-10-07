import numpy as np
from scipy.signal import find_peaks
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
            if name not in ['Left', 'Right']:
                continue
            if "Force" in dev_type or "Plate" in name:
                # Channel 3 is standard vertical Fz in Nexus
                fz, _, _ = vicon.GetDeviceChannel(dev_id, 1, 3)
                fz_array = np.array(fz) * -1
                if total_fz is None:
                    total_fz = fz_array
                else:
                    total_fz += fz_array
    except Exception as e:
        print(f"Error extracting force plate data: {e}")

    if total_fz is None:
        raise RuntimeError("No force plate Fz signal detected in trial.")

    return total_fz, rate

def transform_fp_to_mocap_frame(vicon, fp_frame, fp_rate):
    """Converts a force plate sample index to the corresponding Mocap frame index."""
    if fp_frame is None:
        return None

    mocap_rate = vicon.GetFrameRate()

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

def register_squat_events(vicon, subject, rep_duration: float=1):
    """
    Detects and registers multi-repetition Squat events based on Center of Mass (COM) Z-height:
    - Squat Bottom: Minimum Z-point (peak knee/hip flexion) for each repetition.
    - Squat Start: Frame where COM drops below 98% of standing height prior to bottom.
    - Squat End: Frame where COM returns to 98% of standing height following bottom.
    """
    print("--- Registering Multi-Rep Squat Events ---")

    # 1. Extract Center of Mass Z-trajectory (or Pelvis / LASI proxy if COM output not modeled)
    try:
        # Attempt to pull modeled Center of Mass trajectory
        _, _, z, e = vicon.GetModelOutput(subject, "CentreOfMass")
        com_z = np.array(z)
    except ValueError:
        # Fallback to Pelvis marker (LASI) if CenterOfMass is not present
        try:
            _, _, z1, e = vicon.GetTrajectory(subject, "LASI")
            _, _, z2, e = vicon.GetTrajectory(subject, "RASI")
            _, _, z3, e = vicon.GetTrajectory(subject, "LPSI")
            _, _, z4, e = vicon.GetTrajectory(subject, "RPSI")
            com_z = np.mean([z1, z2, z3, z4], axis=0)
            print("  > Using pelvis marker trajectory as Center of Mass proxy.")
        except Exception as err:
            print(f"Error: Unable to extract COM or Pelvis Z-trajectory: {err}")
            return False

    if com_z is None or len(com_z) == 0:
        print("Error: Empty Z-trajectory array.")
        return False

    # 2. Identify all local minima (bottom of squats)
    # Find peaks on inverted signal (minima on original COM signal)
    min_vals = np.percentile(com_z, 15)
    max_vals = np.max(com_z) * .90
    com_velo = np.gradient(com_z)

    # 3. Process each rep: Calculate 98% standing threshold start and end frames
    mocap_events, cf, rep = {}, 0, 1
    while True:
        # Determine search bounds for standing baseline
        # Bound backward search to either trial start or previous rep's bottom
        try:
            mf = np.where(com_z[cf:] < min_vals)[0][0] + cf
            ff = np.where(com_z[:mf] > max_vals)[0][-1]
            ff = np.where(com_velo[:ff] > 0)[0][-1]
            lf = np.where(com_z[mf:] > max_vals)[0][0] + mf
            lf = np.where(com_velo[lf:] < 0)[0][0] + lf
        except IndexError:
            break
        mocap_events[rep] = {'start': int(ff), 'end': int(lf)}
        cf = lf + 1
        rep += 1

    # 4. Write Repetition Events into Nexus
    _write_events_to_nexus(vicon, subject, mocap_events)
    return True

def register_cmj_events(vicon, subject, threshold_N=20.0):
    """Detects and registers Double-Leg Countermovement Jump (DLCMJ) events."""

    print("--- Registering DLCMJ Events ---")
    fz_signal, fp_frame_rate = get_resultant_fz(vicon)
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
        1: {
            'start':  transform_fp_to_mocap_frame(vicon, start_fp, fp_frame_rate),
            'end': transform_fp_to_mocap_frame(vicon, end_landing_fp, fp_frame_rate),
            'others': {
                'toe_off': transform_fp_to_mocap_frame(vicon, toeoff_fp, fp_frame_rate),
                'impact': transform_fp_to_mocap_frame(vicon, impact_fp, fp_frame_rate),
            }
        }
    }

    # Write events to Nexus
    _write_events_to_nexus(vicon, subject, mocap_events)
    return True

def register_dj_events(vicon, subject, threshold_N=20.0):
    """Detects and registers Drop Jump (DJ) contact and flight phases."""
    print("--- Registering Drop Jump Events ---")
    fz_signal, fps = get_resultant_fz(vicon)

    # Ground contacts (> threshold_N)
    contact_idx = np.where(abs(fz_signal) > threshold_N)[0]
    if len(contact_idx) == 0:
        print("Warning: No force plate contact detected for Drop Jump.")
        return False

    ref_frame = np.where(fz_signal > 200)[0][0]
    impact_1 = np.where(fz_signal[:ref_frame] < threshold_N)[0][-1]
    toe_off = np.where(fz_signal[ref_frame:] < threshold_N)[0][0] + ref_frame

    ref_frame2 = np.where(fz_signal[toe_off:] > 200)[0][0] + toe_off
    impact_2 = np.where(fz_signal[:ref_frame2] < threshold_N)[0][-1]
    end_frame = int(impact_2 + (fps * 2))
    end_frame = end_frame if end_frame < len(fz_signal) else int(len(fz_signal))

    mocap_events = {
        1: {
            'start': transform_fp_to_mocap_frame(vicon, impact_1, fps),
            'end': transform_fp_to_mocap_frame(vicon, end_frame, fps),
            'others': {
                'toe_off': transform_fp_to_mocap_frame(vicon, toe_off, fps),
                'impact': transform_fp_to_mocap_frame(vicon, impact_2, fps),
            }
        }
    }

    _write_events_to_nexus(vicon, subject, mocap_events)
    return True

def register_decel_events(vicon, subject, threshold_N=20.0):
    """Detects and registers Drop Jump (DJ) contact and flight phases."""

    print("--- Registering Drop Jump Events ---")
    fz_signal, fps = get_resultant_fz(vicon)

    # Ground contacts (> threshold_N)
    contact_idx = np.where(abs(fz_signal) > threshold_N)[0]
    if len(contact_idx) == 0:
        print("Warning: No force plate contact detected for Drop Jump.")
        return False

    ref_frame = np.where(fz_signal > 200)[0][0]
    impact = np.where(fz_signal[:ref_frame] < threshold_N)[0][-1]
    toe_off = np.where(fz_signal[ref_frame:] < threshold_N)[0][0] + ref_frame

    com, e = vicon.GetModelOutput(subject, "CentreOfMass")
    com = np.array(com[1])
    com[com==0] = np.nan
    pushoff = np.nanargmax(com)

    mocap_events = {
        1: {
            'start': transform_fp_to_mocap_frame(vicon, impact, fps),
            'end': transform_fp_to_mocap_frame(vicon, toe_off, fps),
            'others': {
                'push': pushoff,
            }
        }
    }

    _write_events_to_nexus(vicon, subject, mocap_events)
    return True

def register_cut_events(vicon, subject, threshold_N=20.0):
    """Detects and registers Drop Jump (DJ) contact and flight phases."""

    print("--- Registering Drop Jump Events ---")
    fz_signal, fps = get_resultant_fz(vicon)

    # Ground contacts (> threshold_N)
    contact_idx = np.where(abs(fz_signal) > threshold_N)[0]
    if len(contact_idx) == 0:
        print("Warning: No force plate contact detected for Drop Jump.")
        return False

    ref_frame = np.where(fz_signal > 200)[0][0]
    impact = np.where(fz_signal[:ref_frame] < threshold_N)[0][-1]
    toe_off = np.where(fz_signal[ref_frame:] < threshold_N)[0][0] + ref_frame

    mocap_events = {
        1: {
            'start': transform_fp_to_mocap_frame(vicon, impact, fps),
            'end': transform_fp_to_mocap_frame(vicon, toe_off, fps),
        }
    }

    _write_events_to_nexus(vicon, subject, mocap_events)
    return True

# ---------------------------------------------------------
# Nexus Writer
# ---------------------------------------------------------
def _write_events_to_nexus(vicon, subject, mocap_events: dict):
    """Clears previous events and creates General + Bilateral events in Nexus."""
    vicon.ClearAllEvents()

    frame_offset = 0.0
    # 1. Write General Context Events
    for rep, rep_data in mocap_events.items():
        vicon.CreateAnEvent(subject, 'General', 'Start', int(rep_data.get('start')), frame_offset)
        vicon.CreateAnEvent(subject, 'General', 'End', int(rep_data.get('end')), frame_offset)
        if 'others' in rep_data:
            other_events = rep_data.get('others')
            for event_name, frame in other_events.items():
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
        if 'SQUAT' in file_name.upper():
            register_squat_events(vicon, subject)
        elif 'CMJ' in file_name.upper():
            register_cmj_events(vicon, subject)
        elif 'DJ' in file_name.upper():
            register_dj_events(vicon, subject)
        elif 'LONGDEC' in file_name.upper():
            register_decel_events(vicon, subject)
        elif ('DECLEFT' in file_name.upper()) or ('DECRIGHT' in file_name.upper()):
            register_decel_events(vicon, subject)
        elif 'CUT' in file_name.upper():
            register_cut_events(vicon, subject)