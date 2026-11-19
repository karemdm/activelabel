@echo off
rem Run visionloop with this folder's virtual environment (Windows). Example: visionloop.cmd serve --open
"%~dp0.venv\Scripts\python.exe" -m visionloop.cli %*
