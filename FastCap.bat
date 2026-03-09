@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "PY_EXE=.venv\Scripts\python.exe"
) else (
  set "PY_EXE=python"
)

set "LOG_DIR=%~dp0logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
for /f %%i in ('powershell -NoProfile -Command "(Get-Date).ToString(\"yyyyMMdd-HHmmss\")"') do set "RUN_STAMP=%%i"
set "LOG_FILE=%LOG_DIR%\fastcap-%RUN_STAMP%.log"

%PY_EXE% -c "import PySide6" >nul 2>&1
if not "%ERRORLEVEL%"=="0" (
  echo.
  echo Missing dependency: PySide6
  echo Install with:
  echo   %PY_EXE% -m pip install PySide6
  echo.
  echo Log file: %LOG_FILE%
  pause
  exit /b 1
)

echo FastCap log file: %LOG_FILE%
%PY_EXE% -X faulthandler "extract_clips_gui.py" --log-file "%LOG_FILE%"

set EXITCODE=%ERRORLEVEL%
if not "%EXITCODE%"=="0" (
  echo.
  echo FastCap app failed with exit code %EXITCODE%.
  echo See log: %LOG_FILE%
  pause
  exit /b %EXITCODE%
)
echo FastCap completed successfully. Log: %LOG_FILE%
exit /b %EXITCODE%
