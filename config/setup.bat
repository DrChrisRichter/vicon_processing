@echo off
setlocal

rem Set the Python executable path
set "python_exe=C:\Python39\python.exe"

rem Run the Python script and capture the output
for /f "delims=" %%i in ('"%python_exe%" "find_nexus_version.py"') do set "nexus_version=%%i"

rem Write the Nexus version to the config file
echo NexusVersion=%nexus_version% > config.txt

endlocal