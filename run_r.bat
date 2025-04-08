@echo off
setlocal enabledelayedexpansion

:: Set path to R script
set "R_SCRIPT=UAV_lidar_dataprocessing_edited.R"

:: Set directory containing the LAS files
set "INPUT_DIR=data\rolleston_forest_plots"

:: Full path to Rscript.exe
set "RSCRIPT_PATH=C:\Program Files\R\R-4.4.2\bin\Rscript.exe"

:: Loop through all .las files in the input directory
for %%f in ("%INPUT_DIR%\*.las") do (
    :: Extract filename only (e.g., plot_1.las)
    set "FILENAME=%%~nxf"
    echo Processing !FILENAME!...

    :: Call R script with: 1) filename, 2) directory path
    "%RSCRIPT_PATH%" "%R_SCRIPT%" "!FILENAME!" "%INPUT_DIR%"
)

echo All files processed.
pause
