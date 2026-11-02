@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
set "PYTHON=%PHASE_PYTHON%"
if not defined PYTHON set "PYTHON=C:\ProgramData\anaconda3\envs\py1\python.exe"
pushd "%PROJECT_ROOT%" >nul
"%PYTHON%" -m phase_agent.runtime.codex_mcp %*
set "EXIT_CODE=%ERRORLEVEL%"
popd >nul
exit /b %EXIT_CODE%
