@echo off
setlocal

set "ROOT=%~dp0"
pushd "%ROOT%"

if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)

%PYTHON% scripts\run_infer.py --text "你好，这是一次本地验证。" --out output\smoke.wav
set "EXITCODE=%ERRORLEVEL%"
popd
exit /b %EXITCODE%
