class Strategy:
    def __init__(self, config):
        self.config = self.load_strategy_config(config['strategy'])
        self.balance = 0
        self.position = {'LONG': {}, 'SHORT': {}}
        self.open_orders = {'LONG': [], 'SHORT': []}
        self.qty_step = None
        self.price_step = None

    def update_steps(self, qty_step, price_step):
        self.qty_step = qty_step
        self.price_step = price_step

    def update_balance(self, balance):
        self.balance = balance

    def update_position(self, position):
        self.position = position

    def update_orders(self, orders):
        self.open_orders = orders

    def update_values(self, balance, position, orders):
        self.update_balance(balance)
        self.update_position(position)
        self.update_orders(orders)

    def make_decision(self, balance, position, orders, price) -> dict:
        changed_orders = {'cancel': [],
                          'add': []}
        return changed_orders

    def on_update(self, position, last_filled_order):
        changed_orders = {'cancel': [],
                          'add': []}
        return changed_orders

    def load_strategy_config(self, config: dict) -> dict:
        return config
