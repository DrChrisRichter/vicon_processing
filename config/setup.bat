@echo off
setlocal

rem 1. Path to your Miniconda activation script and target environment name
set "conda_activate=C:\Users\%USERNAME%\miniconda3\condabin\activate.bat"
set "env_name=vicon_env"

rem 2. Path to your target Miniconda environment Python executable
set "python_exe=C:\Users\%USERNAME%\miniconda3\envs\%env_name%\python.exe"

rem 3. Activate the Conda environment
if exist "%conda_activate%" (
    call "%conda_activate%" %env_name%
) else (
    echo Warning: Conda activate.bat not found. Falling back to direct executable call.
)

rem 4. Run the Python script and capture the output
for /f "delims=" %%i in ('"%python_exe%" "find_nexus_version.py"') do set "nexus_version=%%i"

rem 5. Write the Nexus version to the config file
echo NexusVersion=%nexus_version% > config.txt

endlocal