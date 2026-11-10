@echo off
setlocal
cd /d "%~dp0"
py -3.10 -m streamlit run app.py
pause
