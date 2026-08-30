@echo off
cd /d "%~dp0"
set TF_CPP_MIN_LOG_LEVEL=3
set TF_ENABLE_ONEDNN_OPTS=0
".venv\Scripts\python.exe" "cnn_baseline\cnn_baseline.py"
pause
