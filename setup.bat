@echo off
setlocal

set "ROOT=%~dp0"
pushd "%ROOT%"

if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)

echo [setup] using %PYTHON%
%PYTHON% -m pip install --upgrade pip
if errorlevel 1 goto :end
%PYTHON% -m pip install -r requirements.txt
if errorlevel 1 goto :end

echo [setup] done

:end
set "EXITCODE=%ERRORLEVEL%"
popd
exit /b %EXITCODE%
