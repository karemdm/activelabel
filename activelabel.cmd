@echo off
rem Run activelabel with this folder's virtual environment (Windows). Example: activelabel.cmd serve --open
"%~dp0.venv\Scripts\python.exe" -m activelabel.cli %*
