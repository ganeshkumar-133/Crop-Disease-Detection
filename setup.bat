@echo off
setlocal
cd /d "%~dp0"
echo ==================================================
echo CropShield AI - setup
 echo This installs Python packages. Use Python 3.10.
echo ==================================================
py -3.10 --version
if errorlevel 1 (
  echo Python 3.10 was not found. Install Python 3.10 from python.org, then retry.
  pause
  exit /b 1
)
py -3.10 -m pip install --upgrade pip
if errorlevel 1 goto failed
py -3.10 -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo.
echo Setup complete. Double-click run.bat to open the website.
pause
exit /b 0
:failed
echo.
echo Setup failed. Copy the full error text and send it to ChatGPT.
pause
exit /b 1
