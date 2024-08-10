from optimize_forager import main as optimize
from tools.extract_best_config_forager_mod import main as extract
import sys
import asyncio
import shutil
import os


async def gg():
    while True:
        print(1)
        await asyncio.sleep(1)


def move(source_dir, destination_dir):
    # Создаем папку old, если она не существует
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Копируем файлы
    for item in os.listdir(source_dir):
        source_path = os.path.join(source_dir, item)
        destination_path = os.path.join(destination_dir, item)

        # Пропускаем папку 'old'
        if os.path.isdir(source_path) and item == "old":
            continue

        # Копируем файлы и папки
        if os.path.isfile(source_path):
            shutil.move(source_path, destination_path)


sys.argv = ["config_path", r"C:\pbgui\passivbot\configs\binance.json"]
# Указываем путь к папке
source_dir = "opt_results_forager"
destination_dir = os.path.join(source_dir, "old")
move(source_dir, destination_dir)
# asyncio.run(optimize())
config = os.listdir(source_dir)[0]
if config == "old":
    raise Exception()
sys.argv = [
    "file_location",
    os.path.join(source_dir, config),
    "user",
    "bybit_01",
    "verbose",
]
source_dir = "opt_results_forager_analysis"
destination_dir = os.path.join(source_dir, "old")
move(source_dir, destination_dir)
