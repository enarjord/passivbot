
# boucler sur les configs
# Définir le COIN
SYMBOL_NAME="DOGEUSDT"

for i in configs/live/tests/*.json; 
do  

    live_config="${i}"


    backtest_config="configs/backtest/default.hjson"
    nb_best_coin="1"
    starting_balance="1000.0"
    total_wallet_exposure="1"
    start_date="2021-01-01"
    end_date="2022-06-23"
    user="bybit_tedy" #bybit_tedy sawyer bybit_pro

    # Calculate the 1 bot exposure
    bot_wallet_exposure=$(python3<<<"print(${total_wallet_exposure} / ${nb_best_coin})")
    # Calculate the amount traded by bot
    amount_traded_by_bot=$(python3<<<"print(${total_wallet_exposure} * ${starting_balance} / ${nb_best_coin})")


    sed -r -i "s/starting_balance[ ]*:[ ]*[0-9.]+/starting_balance: ${starting_balance}/g" ${backtest_config}
    sed -r -i "s/\"wallet_exposure_limit\"[ ]*:[ ]*[0-9.]+/\"wallet_exposure_limit\": ${bot_wallet_exposure}/g" ${live_config}
    sed -r -i "s/start_date[ ]*:[ ]*[0-9-]+/start_date: ${start_date}/g" ${backtest_config}
    sed -r -i "s/end_date[ ]*:[ ]*[0-9-]+/end_date: ${end_date}/g" ${backtest_config}
    sed -r -i "s/user[ ]*:[ ]*[a-zA-Z0-9_-]+/user: ${user}/g" ${backtest_config}



    python3 -u backtest.py -s ${SYMBOL_NAME} -oh  ${live_config} | tee "${live_config}_${SYMBOL_NAME}.result.txt"

done;

# For XRP
#best : ./configs/live/tests/long_static_AU_NObtp.json_XRPUSDT.result.txt => 183%
#second best : ./configs/live/tests/long_recursive_NOAU_btp.json_XRPUSDT.result.txt => 152%
find ./configs/live/tests/ -name "*XRP*.txt" -print -exec grep "\(Total gain\|noneeded\)" {} \;

# For DOGE
#best : ./configs/live/tests/long_static_AU_btp.json_DOGEUSDT.result.txt => 735%
#second best : ./configs/live/tests/long_static_AU_NObtp.json_DOGEUSDT.result.txt => 530%
find ./configs/live/tests/ -name "*DOGE*.txt" -print -exec grep "\(Total gain\|noneeded\)" {} \;
