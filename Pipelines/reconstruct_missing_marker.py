from FileProcessor import ReconstructMissingMarker

import os
import datetime
from viconnexusapi import ViconNexus

# Connect to Nexus
vicon = ViconNexus.ViconNexus()
assert len(vicon.GetSubjectNames()) == 1, 'More then One subject in the file'
subject = vicon.GetSubjectNames()[0]
filepath = vicon.GetTrialName()[0]
filename = vicon.GetTrialName()[1]
session = os.sep.join(filepath.split(os.sep)[-2:-1])
x1d_file = os.path.join(filepath, filename) + '.x1d'
if not os.path.exists(x1d_file):
    raise FileNotFoundError("No x1d file found")
capture_date = datetime.datetime.fromtimestamp(os.path.getctime(x1d_file)).strftime("%Y-%m-%d")

# check current directory
cwd = os.getcwd()
cwd = os.path.join(cwd.split('BioApp-by-Aspetar')[0], 'BioApp-by-Aspetar')
os.chdir(cwd)

# get all measures
print("Started Process")

ReconstructMissingMarker.estimate_missing_marker(
    api=vicon, subject_name=subject, file_path=filepath, file_name=filename + '.c3d'
)

print("Done")