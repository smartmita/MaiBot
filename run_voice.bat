@echo off
start "Voice Adapter" cmd /k "call conda activate maicore && cd /d C:\GitHub\maimbot_tts_adapter && echo Running Napcat Adapter... && python maimbot_pipeline.py" 