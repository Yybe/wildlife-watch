@echo off
rem Starts the Wildlife Watch demo server. Double-click this file, then open the URL shown.
cd /d "%~dp0.."
echo Starting Wildlife Watch demo... the browser opens automatically when the model is ready.
start /b "" cmd /c "timeout /t 8 /nobreak >nul & start http://127.0.0.1:8765"
..\.venv\Scripts\python.exe demo\server.py
pause