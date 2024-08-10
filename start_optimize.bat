call C:\pbgui\passivbot\update.bat
cd C:\pbgui\passivbot
C:\pbgui\venv\Scripts\python.exe C:\pbgui\passivbot\src\optimize_forager.py "C:\pbgui\passivbot\configs\binance.json" 1> "C:\pbgui\passivbot\optimize.log" 2>&1
curl "https://notify.dro4illa.ru?n=Passivbot&s=optimize-done"