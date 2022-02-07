
bash_symbols = ['SXPUSDT','CELRUSDT','CHRUSDT','DOGEUSDT','CHZUSDT','ONEUSDT','1INCHUSDT',
 'ENJUSDT', 'RSRUSDT', 'GRTUSDT', 'VETUSDT'] #ATTENTION IL FAUT LES virgules

################### shell script generating ##############
print('---------------------------------------Step 3 : generate shell scripts-------------------------------------------------')
print ("Linux Bash to create the screens commands in run_server_live.sh")
file_content = ""
file_content += "#!/bin/bash"+"\n"
file_content += 'symbols=('
for symbol in bash_symbols:
    file_content +=   symbol + " "
file_content += ')'+"\n"
file_content += 'for i in "${symbols[@]}"'+"\n"
file_content += 'do'+"\n"
file_content += '    :'+"\n"
file_content += '    echo "Running screen on $i"'+"\n"
file_content += '    screen -S "tedy_$i" -dm bash -c "cd /home/tedy/Documents/passivbot5.3/passivbot;python3 passivbot.py bybit_tedy $i  configs/live/auto_unstuck_enabled.example.json"'+"\n"
file_content += 'done'+"\n"
file_content += ''+"\n"

file = open('run_server_live.sh', 'w')
file.write(file_content)
file.close()

print('----------------------------------------------------------------------------------------')
print ("Linux Bash to kill all screens (stop_server_live.sh) ")
file_content = ""
file_content += "#!/bin/bash"+"\n"
file_content += 'symbols=('
for symbol in bash_symbols:
    file_content +=   symbol + " "
file_content += ')'+"\n"
file_content += 'for i in "${symbols[@]}"'+"\n"
file_content += 'do'+"\n"
file_content += '    :'+"\n"
file_content += '    echo "Kill screen for $i"'+"\n"
file_content += '    screen -S "tedy_$i" -X quit'+"\n"
file_content += 'done'+"\n"
file_content += ''+"\n"


file = open('stop_server_live.sh', 'w')
file.write(file_content)
file.close()
