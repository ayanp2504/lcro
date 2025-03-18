@echo off 
call "C:\Excellerate LLM\App\.env\Scripts\activate.bat" 
cd /d "C:\Excellerate LLM\App\backend_lcro" 
"C:\Excellerate LLM\App\.env\Scripts\uvicorn.exe" backend_lcro.app:app --host 172.31.11.168 --port 90 
