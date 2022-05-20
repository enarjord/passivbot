import argparse
import os
import hjson

def arguments_management():
    ### Parameters management
    parser = argparse.ArgumentParser( description="This script will create Shell script for my server",
    epilog="",
    usage="python3 " + __file__ + " user_keys -jf tmp/grid_ok_coins.json tedy configs/live/a_tedy.json"
    )

    parser.add_argument("user_name", type=str, help="api_key username")
    parser.add_argument("live_config_filepath", type=str, help="file path to live config")

    parser.add_argument("-jf","--json-file-coin-list",
                        type=str,required=False,dest="json_file_coin_list",default="",
                        help="A json file containing a coin list array, ex : tmp/grid_ok_coins.json",
    )

    parser.add_argument("-cl","--coin_list",
                        type=str,required=False,dest="coin_list",default="",
                        help="A list of coin separated by space, ex : 'ONEUSDT XLMUSDT'",
    )

    parser.add_argument("-type","--type",
                        type=str,required=False,dest="type",default="futures",
                        help="futures or spot",
    )

    args = parser.parse_args()


    if not os.path.exists(args.live_config_filepath) :
        print("live_config_path doesn't exist")
        exit()

    # print(args.live_config_filepath)     
    # print(os.path.dirname(os.path.abspath(__file__))+'/')     
    base_dir = os.path.realpath(os.path.dirname(os.path.abspath(__file__))+'/../')+'/'
    args.live_config_filepath       = (os.path.realpath(args.live_config_filepath)).replace(base_dir, '')
    # print(args.live_config_filepath)     


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

sport_part = ""
if args.type == 'spot':
    sport_part = " -m spot "

################### shell script generating ##############
print('---------------------------------------Step 3 : generate shell scripts-------------------------------------------------')
print ("Linux Bash to create the screens commands in run_server_live_" + args.user_name + ".sh")
file_content = ""
file_content += "#!/bin/bash"+"\n"

file_content += "current_pwd=`pwd`"+"\n"
file_content += "gs=' -gs '"+"\n"
file_content += "gs=''"+"\n"
file_content += 'symbols=('
for symbol in bash_symbols:
    file_content +=   symbol + " "
file_content += ')'+"\n"
file_content += 'for i in "${symbols[@]}"'+"\n"
file_content += 'do'+"\n"
file_content += '    :'+"\n"
file_content += '    echo "Running screen on $i"'+"\n"
file_content += '    screen -S "' + args.user_name + '_$i" -dm bash -c "cd ${current_pwd}/;python3 passivbot.py $gs ' + args.user_name + ' $i  ' + args.live_config_filepath + ' ' + sport_part + '"'+"\n"
file_content += 'done'+"\n"
file_content += ''+"\n"

file = open('../run_server_live_' + args.user_name + '.sh', 'w')
file.write(file_content)
file.close()

print('----------------------------------------------------------------------------------------')
print ("Linux Bash to kill all screens (stop_server_live_" + args.user_name + ".sh) ")
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
file_content += '    screen -S "' + args.user_name + '_$i" -X quit'+"\n"
file_content += 'done'+"\n"
file_content += ''+"\n"


file = open('../stop_server_live_' + args.user_name + '.sh', 'w')
file.write(file_content)
file.close()
