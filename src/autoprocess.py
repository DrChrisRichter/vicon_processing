from pathlib import Path
import numpy as np
import ezc3d
import os

from src import enf_reader
from viconnexusapi import ViconNexus

# 1. Initialize the connection to the running Vicon Nexus application
vicon = ViconNexus.ViconNexus()

# Option 2: Forward slashes everywhere (Windows Python understands this perfectly)
for subject_path in Path('C:/vicontest/SSCData').iterdir():
    # get all files
    files = [x for x in subject_path.glob('*/*.x1d')]
    # check for static
    try:
        static_path = [x for x in files if 'static' in x.name.lower()][0]
    except IndexError:
        print(f'{subject_path}: No static data')
        continue
    tasks_path = [x for x in files if 'static' not in x.name.lower()]
    for file in tasks_path:

        # check enf if force plate are activated
        enf_file = sorted(list(file.parent.glob(f"{file.with_suffix('').name}*.enf")))[-1]
        enf_data = enf_reader.read_enf(enf_file.as_posix())

        if 'NOTES' in enf_data['TRIAL_INFO']:
            if enf_data['TRIAL_INFO']['NOTES'] == 'BAIOPipe:OK':
                continue
            elif enf_data['TRIAL_INFO']['NOTES'] in ['STATUS:ModelFile', 'STATUS:OK']:
                continue
            elif enf_data['TRIAL_INFO']['NOTES'] == '':
                pass
            elif 'STATUS:Check' in enf_data['TRIAL_INFO']['NOTES']:
                pass
            else:
                hu = 5

        if 'DESCRIPTION' in enf_data['TRIAL_INFO']:
            if enf_data['TRIAL_INFO']['DESCRIPTION'] == 'BAIOPipe:OK':
                continue
            elif enf_data['TRIAL_INFO']['DESCRIPTION'] in ['NS']:
                continue
            elif 'PC' in enf_data['TRIAL_INFO']['DESCRIPTION']:
                continue

        if 'FP1' not in enf_data['TRIAL_INFO'] or 'FP2' not in enf_data['TRIAL_INFO']:
            enf_data['TRIAL_INFO']['FP1'] = 'Auto'
            enf_data['TRIAL_INFO']['FP2'] = 'Auto'
            enf_reader.write_enf(enf_data, enf_file.as_posix())
        else:
            if (enf_data['TRIAL_INFO']['FP1'] == 'Invalid' and
                    enf_data['TRIAL_INFO']['FP2'] == 'Invalid'):
                enf_data['TRIAL_INFO']['FP1'] = 'Auto'
                enf_data['TRIAL_INFO']['FP2'] = 'Auto'
                enf_reader.write_enf(enf_data, enf_file.as_posix())

        # 3. Open the trial (Path, Timeout in milliseconds)
        timeout_ms = 60000
        vicon.OpenTrial(str(file.with_suffix('')), timeout_ms)

        # 3. Run the pipeline (Name, Timeout in milliseconds)
        pipeline_name = 'SSC Filter + Dynamic PiG (Reports Only)'
        print(f"{file.with_suffix('')} > Running pipeline...")
        # 2. Pass an empty string for the MacroPath/MacroName
        macro_path = ""
        # 4. Call RunPipeline with all three arguments
        ans = vicon.RunPipeline(pipeline_name, macro_path, timeout_ms)

        enf_data['TRIAL_INFO']['NOTES'] = 'BAIOPipe:OK'
        enf_reader.write_enf(enf_data, enf_file.as_posix())