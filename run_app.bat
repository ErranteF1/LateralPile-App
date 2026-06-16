@echo off
REM --- Lateral Pile Analysis app launcher (Windows) ---
cd /d "%~dp0"
python -m streamlit run app.py
pause
