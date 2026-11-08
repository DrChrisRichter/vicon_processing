from pathlib import Path
import time

from src import enf_reader
from Pipelines import gap_fill_pipeline
from viconnexusapi import ViconNexus

# 1. Initialize the connection to the running Vicon Nexus application
vicon = ViconNexus.ViconNexus()

# Option 2: Forward slashes everywhere (Windows Python understands this perfectly)
successful = 'baio:Cleaned&ModelledV2.01'
timeout_ms = 60000

for subject_path in Path('C:/vicontest/SSCData').iterdir():

    if not subject_path.is_dir():
        continue

    if str(subject_path.name) not in ['JV 368303 T2', 'KD 087108 T2', 'KO 356779 T2', 'LB 368624 T2',
                                  'LH 370790 T2', 'MB 360840 T2 Retest', 'MC 367232 T2', 'KM 354905']:
        continue
    # get all files
    files = [x for x in subject_path.glob('*/*.x1d')]
    # check for static
    try:
        static_path = [
            x for x in files
            if 'static' in x.name.lower()
               and 'pose' not in x.name.lower()
               and x.with_suffix('.c3d').exists()
        ][-1]
    except IndexError:
        print(f'{subject_path}: No static data')
        continue
    tasks_path = [x for x in files if 'static' not in x.name.lower()]
    for file in tasks_path:

        if file.with_suffix('').name.startswith('._'):
            continue

        # check enf if force plate are activated
        enf_file = sorted(list(file.parent.glob(f"{file.with_suffix('').name}*.enf")))[-1]
        enf_data = enf_reader.read_enf(enf_file.as_posix())

        if 'NOTES' in enf_data['TRIAL_INFO']:
            if enf_data['TRIAL_INFO']['NOTES'] == successful:
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
            if enf_data['TRIAL_INFO']['DESCRIPTION'] == successful:
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
        vicon.OpenTrial(str(file.with_suffix('')), timeout_ms)

        # 3. Run the pipeline (Name, Timeout in milliseconds)
        pipeline_name = 'Reconstruct + Filter + Dynamic PiG'
        print(f"{file.with_suffix('')} > Running pipeline...")
        # 2. Pass an empty string for the MacroPath/MacroName
        macro_path = ""
        # 4. Call RunPipeline with all three arguments
        vicon.SetTrialRegionOfInterest(1, vicon.GetFrameCount()-1)
        time.sleep(2)
        vicon.RunPipeline('baio_Reconstruct', macro_path, timeout_ms)
        time.sleep(2)
        vicon.RunPipeline('baio_Label', macro_path, timeout_ms)
        time.sleep(2)
        vicon.RunPipeline('baio_FillGaps', macro_path, timeout_ms)
        time.sleep(2)
        vicon.RunPipeline('baio_FilterMdlSave', macro_path, timeout_ms)


        # check history file
        #<OperationSubType name="Label" notes="">
        #<Parameters>
        #  <Param header="0" name="Date" value="2026-Nov-04 18:27:12"/>
        #  <Param header="1" name="Frame Range" value=""/>
        #  <Param header="0" name="First Frame" value="1"/>
        #  <Param header="0" name="Last Frame" value="1752"/>
        #  <Param header="1" name="Labeling" value=""/>
        #  <Param header="0" name="Label Completeness Entrance Threshold" value="0.84999999999999998"/>
        #  <Param header="0" name="Label Completeness Exit Threshold" value="0.59999999999999998"/>
        #  <Param header="0" name="Booting Versus Tracking" value="0.5"/>
        #  <Param header="0" name="Tracking Quality" value="0"/>
        #  <Param header="0" name="Smoothing Factor" value="1"/>
        #  <Param header="0" name="Single pass" value="No"/>
        #</Parameters>
        #</OperationSubType>

        # failed = check_pipeline_failure(path=file)
        enf_data['TRIAL_INFO']['NOTES'] =successful
        enf_reader.write_enf(enf_data, enf_file.as_posix())
        # Pauses the script for 2 seconds
        time.sleep(2)
        # close file
        vicon.CloseTrial(timeout_ms)
        # Pauses the script for 2 seconds
        time.sleep(2)
        
    time.sleep(10)