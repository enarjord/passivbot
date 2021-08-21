import subprocess


def main():
    kwargs_list = [
        {'optimize_config_path': 'configs/optimize/scalp.hjson',
         'symbol': 'XMRUSDT',
         'starting_balance': 1000.0},
        {'optimize_config_path': 'configs/optimize/scalp.hjson',
         'symbol': 'BTCUSD_PERP',
         'starting_balance': 0.1},
        {'optimize_config_path': 'configs/optimize/vanilla.hjson',
         'symbol': 'ADABTC',
         'starting_balance': 0.1,
         'market_type': 'spot'},
        {'optimize_config_path': 'configs/optimize/scalp.hjson',
         'user': 'ebybit',
         'symbol': 'EOSUSD',
         'starting_balance': 100.0},
    ]

    for kwargs in kwargs_list:
        formatted = f"python3 optimize.py {kwargs['optimize_config_path']}"
        for key in [k for k in kwargs if k != 'optimize_config_path']:
            formatted += f' --{key} {kwargs[key]}'
        print(formatted)
        subprocess.run([formatted], shell=True)


if __name__ == '__main__':
    main()