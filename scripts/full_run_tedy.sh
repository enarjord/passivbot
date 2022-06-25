
# live_config="../configs/live/a_tedy.json"
live_config="../configs/live/a_tedy.json"
backtest_config="../configs/backtest/default.hjson"
nb_best_coin="3"
starting_balance="1000.0"
total_wallet_exposure="3"
start_date="2021-01-01"
end_date="2022-06-23"
user="bybit_tedy"

echo "-------------------------"
# Update the Starting balance for backtest
sed -r -i "s/starting_balance[ ]*:[ ]*[0-9.]+/starting_balance: ${starting_balance}/g" ${backtest_config}
echo "Starting balance => ${starting_balance}$"

# Calculate the 1 bot exposure
bot_wallet_exposure=$(python3<<<"print(${total_wallet_exposure} / ${nb_best_coin})")
# "wallet_exposure_limit": 0.15
sed -r -i "s/\"wallet_exposure_limit\"[ ]*:[ ]*[0-9.]+/\"wallet_exposure_limit\": ${bot_wallet_exposure}/g" ${live_config}
echo "Bot wallet exposure => ${bot_wallet_exposure}"

# Calculate the amount traded by bot
amount_traded_by_bot=$(python3<<<"print(${total_wallet_exposure} * ${starting_balance} / ${nb_best_coin})")
echo "Amount traded by 1 bot => ${amount_traded_by_bot}$"


# Calculate the amount traded by bot
sed -r -i "s/start_date[ ]*:[ ]*[0-9-]+/start_date: ${start_date}/g" ${backtest_config}
echo "start_date => ${start_date}"
# Calculate the amount traded by bot
sed -r -i "s/end_date[ ]*:[ ]*[0-9-]+/end_date: ${end_date}/g" ${backtest_config}
echo "end_date => ${end_date}"

# set the user
# user: bybit_tedy
sed -r -i "s/user[ ]*:[ ]*[a-zA-Z0-9_-]+/user: ${user}/g" ${backtest_config}
echo "user => ${user}"

echo "Full wallet exposure => ${total_wallet_exposure}$"
echo "Nb best coin => ${nb_best_coin}$"

echo "-------------------------"


echo "Using live config => ${live_config}"
echo "Using backtest config => ${backtest_config}"
echo "Searching nb coins => ${nb_best_coin}"
echo "-------------------------"

read -r -p "Are you sure? [y/N] " response
case "$response" in
    [yY][eE][sS]|[yY]) 
        
        ;;
    *)
        exit
        ;;
esac


#python3 0_coin_grid_validator.py ${live_config} ${backtest_config} -mv24 0 -mt24 0
#python3 1_backtest_looper.py ${live_config} ${backtest_config} -jf tmp/grid_ok_coins.json -oh


# python3 2_backtest_summary.py ${nb_best_coin} ${live_config} ${backtest_config} \
# -min-days 400 \
# -min-gain 100 \
# -max-marketcap-pos 20  
# #-max-stuck-avg 7
# # -max-stuck 200

python3 3_server_script_generator.py  ${user} ${live_config} -jf tmp/best_coins.json