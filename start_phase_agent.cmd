@echo off
setlocal
cd /d "%~dp0"
where conda >nul 2>nul
if %errorlevel%==0 (
  conda run -n py1 --no-capture-output python -m run.local_project_launcher
) else (
  if exist "C:\ProgramData\anaconda3\envs\py1\python.exe" (
    "C:\ProgramData\anaconda3\envs\py1\python.exe" -m run.local_project_launcher
  ) else (
    echo Cannot find py1. Configure conda on PATH or edit this launcher to point to py1.
    pause
  )
)
