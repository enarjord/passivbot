import argparse
import os
import hjson

def arguments_management():
    ### Parameters management
    parser = argparse.ArgumentParser( description="This script will create Shell script for my server",
    epilog="",
    usage="python3 " + __file__ + " -jf tmp/grid_ok_coins.json"
    )
    parser.add_argument("-jf","--json-file-coin-list",
                        type=str,required=False,dest="json_file_coin_list",default="",
                        help="A json file containing a coin list array, ex : tmp/grid_ok_coins.json",
    )

    parser.add_argument("-cl","--coin_list",
                        type=str,required=False,dest="coin_list",default="",
                        help="A list of coin separated by space, ex : 'ONEUSDT XLMUSDT'",
    )

    args = parser.parse_args()

    args.builded_coin_list = []
    if (len(args.coin_list.strip().split(' ')) > 0) :
       args.builded_coin_list = args.coin_list.strip().split(' ')
       if args.builded_coin_list[0] == '' :
           args.builded_coin_list.pop(0)

    if os.path.exists(args.json_file_coin_list) :
        args.builded_coin_list = hjson.load(open(args.json_file_coin_list, encoding="utf-8"))

    if len(args.builded_coin_list) == 0 :
        print('No coin finded with the program arguments. See arguments -jf or -cl')
        exit()

    return args

args = arguments_management()
bash_symbols = args.builded_coin_list

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

file = open('../run_server_live.sh', 'w')
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


file = open('../stop_server_live.sh', 'w')
file.write(file_content)
file.close()
