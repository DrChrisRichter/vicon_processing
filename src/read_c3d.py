import os
import c3d
import numpy as np
import pandas as pd
import configparser
import xml.etree.ElementTree as ET

def read_c3d(file_path, read_mocap=True):
    """
    Reads a C3D file and extracts motion capture and force plate data.

    Args:
        file_path (str): Path to the .c3d file.
        read_mocap (bool): Whether to parse the time-series frame data.

    Returns:
        dict: Contains 'MoCap' (DataFrame), 'GRF' (DataFrame), and 'Info' (dict metadata),
              or an 'Error' key if reading fails.
    """
    if not os.path.exists(file_path):
        return {'Error': 'File does not exist'}

    try:
        with (open(file_path, 'rb') as file_id):
            try:
                reader = c3d.Reader(file_id)
            except Exception:
                raise ValueError('Reading Error of file')

            # 1. Extract Metadata
            info = _extract_metadata(reader, file_path)

            # 2. Extract Labels
            mocap_labels, force_labels = _extract_labels(reader)

            # 3. Extract Force Plate Geometry (+ per-plate mount orientation from .system)
            info = _extract_fp_geometry(reader, info)
            info = _extract_system_orientations(file_path, info)

            # 4. Extract Time-Series Data (if requested)
            if read_mocap:
                mocap_data, force_data = _extract_timeseries_data(reader, info, mocap_labels, force_labels)
            else:
                mocap_data = []
                force_data = []

            return {'MoCap': mocap_data, 'GRF': force_data, 'Info': info}

    except OSError as err:
        return {'Error': f'{err}'}
    except ValueError as err:
        return {'Error': f'{err}'}
    except AttributeError as err:
        return {'Error': f'{err}'}
    except Exception as err:
        return {'Error': f'Previously undiscovered error {file_path}'}


# ---------------------------------------------------------
# Private Helper Functions
# ---------------------------------------------------------

def _extract_metadata(reader, file_path):
    """
    Extracts trial, subject, and processing info.
    Attempts to read from the C3D header, but overwrites with
    data from .history and .Trial.enf sidecar files if they exist.
    """
    info = {}

    # Preserve Vicon's own point classifications. These groups distinguish physical marker
    # positions from modeled points and vector-valued Nexus outputs which all occupy C3D's
    # generic POINT storage.
    point_info = reader.get('POINT')
    point_groups = {}
    for kind, labels_key, units_key in (
        ('angle', 'ANGLES', 'ANGLE_UNITS'),
        ('force', 'FORCES', 'FORCE_UNITS'),
        ('moment', 'MOMENTS', 'MOMENT_UNITS'),
        ('power', 'POWERS', 'POWER_UNITS'),
        ('model_marker', 'MODELED_MARKERS', 'MODELED_MARKER_UNITS'),
    ):
        labels_param = point_info.get(labels_key) if point_info else None
        units_param = point_info.get(units_key) if point_info else None
        labels = [str(value).strip().replace(' ', '') for value in (
            labels_param.string_array if labels_param is not None else []
        )]
        units = [str(value).strip() for value in (
            units_param.string_array if units_param is not None else []
        )]
        if labels:
            point_groups[kind] = {
                'labels': labels,
                'unit': units[0] if units else '',
            }
    point_units = point_info.get('UNITS') if point_info else None
    units = [str(value).strip() for value in (
        point_units.string_array if point_units is not None else []
    )]
    info['POINT_GROUPS'] = point_groups
    info['POINT_UNIT'] = units[0] if units else 'mm'

    # 1. Extract standard processing info from C3D (Mass, Height, Rates)
    processing_info = reader.get('PROCESSING')
    col_of_int_proc = ['BODYMASS', 'HEIGHT']
    for field in col_of_int_proc:
        info[field] = None
        if processing_info:
            val = processing_info.get(field)
            if val and val.dimensions[0] != 0:
                info[field] = val.float_value

    subjects_info = reader.get('SUBJECTS')
    info['Names'] = subjects_info.get('Names').float_value if subjects_info and subjects_info.get('Names') else None

    trial_info = reader.get('TRIAL')
    val = trial_info.get('CAMERA_RATE')
    info['CAMERA_RATE'] = val.float_value if val else None

    analog_info = reader.get('ANALOG')
    info['FP_RATE'] = analog_info.get('RATE').float_value if analog_info else None

    # 2. Initialize string fields in the source metadata schema.
    info['DATEOFCAPTURE'] = [None]
    info['NOTE'] = [None]
    info['DESCRIPTION'] = [None]

    # 3. Augment with sidecar files (.history and .Trial.enf)
    history_paths = [
        file_path.replace(".c3d", ".history.xml"),
        file_path.replace(".c3d", ".history"),
    ]
    enf_path = file_path.replace(".c3d", ".Trial.enf")

    # Read Capture Date from XML (.history)
    history_path = next((path for path in history_paths if os.path.exists(path)), None)
    if history_path:
        try:
            tree = ET.parse(history_path)
            start_date_element = tree.getroot().find(".//Param[@name='Capture Date']")
            if start_date_element is not None:
                info['DATEOFCAPTURE'] = [start_date_element.get('value')]
        except Exception as e:
            print(f"Warning: Failed to parse history file {history_path}: {e}")

    # Read Notes and Description from INI (.Trial.enf)
    if os.path.exists(enf_path):
        try:
            config = configparser.ConfigParser()
            config.read(enf_path)
            if config.has_section('TRIAL_INFO'):
                note = config.get('TRIAL_INFO', 'NOTES', fallback=None)
                desc = config.get('TRIAL_INFO', 'DESCRIPTION', fallback=None)
                if note: info['NOTE'] = [note]
                if desc: info['DESCRIPTION'] = [desc]
                assignments = {}
                for key, value in config.items('TRIAL_INFO'):
                    if key.upper().startswith('FP') and key[2:].isdigit():
                        side = value.strip().lower()
                        assignments[key[2:]] = {'left': 'l', 'right': 'r'}.get(side, side)
                if assignments:
                    info['FP_ASSIGNMENTS'] = assignments
        except Exception as e:
            print(f"Warning: Failed to parse ENF file {enf_path}: {e}")

    return info


def _extract_labels(reader):
    """Formats and returns lists of MoCap and Force Plate labels."""
    try:
        mocap_labels = [x.replace(' ', '') for x in reader.point_labels]
    except AttributeError:
        raise AttributeError("file has no reconstructed markers and should not be processed")
    mocap_labels = sum([[f"{x}_x", f"{x}_y", f"{x}_z"] for x in mocap_labels], [])

    force_labels = reader.get('ANALOG')
    if force_labels:
        force_labels = force_labels.get('LABELS').string_array
        force_labels = [x.replace(' ', '').replace('Force.', '').replace('Moment.', '') for x in force_labels]
    else:
        force_labels = []

    return mocap_labels, force_labels


def _extract_fp_geometry(reader, info):
    """Extracts Force Plate dimensions and calculates mid-points."""
    platform = reader.get('FORCE_PLATFORM')
    corners = platform.get('CORNERS') if platform else None
    if corners is None:
        return info
    fp_border = corners.float_array
    n_planes, n_points, n_plates = fp_border.shape

    raw = np.array(fp_border)
    fp_border = raw.reshape(n_plates, n_points, n_planes)
    for count in range(n_plates):
        info[f'xmidposFP{count + 1}'] = float(fp_border[count].mean(axis=0)[0])

    # Full corner geometry (mm, lab frame) for the 3D viewer. c3d's float_array already
    # returns CORNERS as (plate, corner, coord) — e.g. (4, 4, 3) — so x/y/z stay grouped
    # per corner. Use it as-is; the reshape above is kept only for xmidpos compatibility.
    info['FP_CORNERS'] = raw.tolist()

    return info


def _extract_system_orientations(file_path, info):
    """Read per-plate mount orientation from the Vicon `.system` sidecar (XML), if present.

    Each force-plate device carries `StandardPosition_{X,Y}` (plate centre, metres) and
    `StandardOrientation_Z` (mount yaw about vertical, radians). We store them as
    `info['FP_ORIENTATIONS'] = [{'x','y','yaw'}, ...]`; the analytics matches each c3d plate to
    the nearest one by position (the .system device order does NOT match the c3d plate order).
    Absent sidecar → key left unset and the caller falls back to its default yaw.
    """
    sys_path = file_path.replace('.c3d', '.system')
    if not os.path.exists(sys_path):
        return info
    try:
        root = ET.parse(sys_path).getroot()
        plates = []
        for pl in root.iter('ParamList'):
            params = {p.get('name'): p.get('value') for p in pl.findall('Param')}
            if params.get('DeviceCategory') != 'Force Plate':
                continue
            try:
                plates.append({
                    'x': float(params['StandardPosition_X']),
                    'y': float(params['StandardPosition_Y']),
                    'yaw': float(params.get('StandardOrientation_Z') or 0.0),
                })
            except (KeyError, TypeError, ValueError):
                continue
        if plates:
            info['FP_ORIENTATIONS'] = plates
    except Exception as e:
        print(f"Warning: Failed to parse system file {sys_path}: {e}")
    return info


def _extract_timeseries_data(reader, info, mocap_labels, force_labels):
    """Iterates through C3D frames to build MoCap and GRF Pandas DataFrames."""
    frames = reader.header.last_frame - reader.header.first_frame + 1
    frequency_ratio = int(info['FP_RATE'] / info['CAMERA_RATE']) if force_labels else 0

    mocap_data = np.empty(shape=(frames, len(mocap_labels)))
    force_frames = int(frames * frequency_ratio)
    force_data = np.empty(shape=(force_frames, len(force_labels)))

    force_frame_indices = [int(x) for x in np.linspace(0, force_frames, num=frames + 1)]
    current_frame = 0

    # Read data frame by frame
    for frame, points, analog in reader.read_frames(copy=False):
        if frame < reader.header.first_frame or frame > reader.header.last_frame:
            continue
        mocap_data[current_frame, :] = np.array(points[:, 0:3]).reshape(1, len(mocap_labels))

        start_idx = force_frame_indices[current_frame]
        end_idx = force_frame_indices[current_frame + 1]
        for channel in range(0, len(force_labels)):
            force_data[start_idx:end_idx, channel] = np.array(analog[channel])

        current_frame += 1

    # Define MoCap DataFrame
    frame_rate = (1 / info['CAMERA_RATE'])
    n_frames = reader.last_frame - reader.first_frame + 1

    mocap_times = np.linspace(reader.first_frame, reader.last_frame, num=n_frames)
    mocap_times = (mocap_times - 1) * frame_rate
    mocap_df = pd.DataFrame(mocap_data, columns=mocap_labels, index=mocap_times)

    if not force_labels:
        return mocap_df, pd.DataFrame()

    # Define Force DataFrame
    ratio = int(reader.analog_rate / reader.point_rate)
    force_times = np.linspace(reader.first_frame, reader.last_frame + 1 - (1 / ratio), num=n_frames * ratio)
    force_times = (force_times * 10).round() / 10
    force_times = (force_times - 1) * frame_rate
    force_df = pd.DataFrame(force_data, columns=force_labels, index=force_times)

    # Invert the plates like Vicon would do
    labels_to_invert, label_weights = [], []
    for col in ['Fx1', 'Fy1', 'Fz1', 'Fx2', 'Fy2', 'Fz2']:
        if col in force_df.columns:
            labels_to_invert.append(col)
            label_weights.append(-1)

    if labels_to_invert:
        force_df[labels_to_invert] = force_df[labels_to_invert] * label_weights

    return mocap_df, force_df