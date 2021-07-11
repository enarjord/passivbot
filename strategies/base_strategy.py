from typing import List, Tuple

from definitions.order import Order
from definitions.order_list import OrderList, empty_order_list
from definitions.position_list import PositionList


class Strategy:
    def __init__(self, config):
        self.config = self.load_strategy_config(config['strategy'])
        self.balance = 0
        self.position = PositionList()
        self.open_orders = OrderList()
        self.qty_step = None
        self.price_step = None

    def precompile(self):
        pass

    def update_steps(self, qty_step, price_step):
        self.qty_step = qty_step
        self.price_step = price_step

    def update_balance(self, balance: float):
        self.balance = balance

    def update_position(self, position: PositionList):
        self.position.update_long(position.long)
        self.position.update_short(position.short)

    def update_orders(self, orders: OrderList):
        self.open_orders.update_long(orders.long)
        self.open_orders.update_short(orders.short)

    def update_values(self, balance: float, position: PositionList, orders: OrderList):
        self.update_balance(balance)
        self.update_position(position)
        self.update_orders(orders)

    def make_decision(self, balance: float, position: PositionList, orders: OrderList, price: float) -> Tuple[
        List[Order], List[Order]]:
        add_orders = empty_order_list()
        delete_orders = empty_order_list()
        return add_orders, delete_orders

    def on_update(self, position: PositionList, last_filled_order: Order) -> Tuple[List[Order], List[Order]]:
        add_orders = empty_order_list()
        delete_orders = empty_order_list()
        return add_orders, delete_orders

    def load_strategy_config(self, config: dict) -> dict:
        return config
