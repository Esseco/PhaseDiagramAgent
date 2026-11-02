@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
set "PYTHON=%PHASE_PYTHON%"
if not defined PYTHON set "PYTHON=C:\ProgramData\anaconda3\envs\py1\python.exe"
pushd "%PROJECT_ROOT%" >nul
if /I "%~1"=="start" goto lifecycle
if /I "%~1"=="stop" goto lifecycle
if /I "%~1"=="takeover" goto lifecycle
if /I "%~1"=="mcp" goto mcp
"%PYTHON%" -m phase_agent.runtime.codex_control %*
set "EXIT_CODE=%ERRORLEVEL%"
goto done
:mcp
shift
set "MCP_ARGS="
:collect_mcp_args
if "%~1"=="" goto run_mcp
set "MCP_ARGS=%MCP_ARGS% "%~1""
shift
goto collect_mcp_args
:run_mcp
"%PYTHON%" -m phase_agent.runtime.codex_mcp %MCP_ARGS%
set "EXIT_CODE=%ERRORLEVEL%"
goto done
:lifecycle
"%PYTHON%" -m phase_agent.runtime.codex_agent %*
set "EXIT_CODE=%ERRORLEVEL%"
:done
popd >nul
exit /b %EXIT_CODE%
