@echo off
chcp 65001 > nul
cd /d "%~dp0"
python -m streamlit run app.py --server.port 8517
pause
