from numba import types, typed, typeof, njit
from numba.experimental import jitclass

from definitions.order import Order


@njit
def empty_order_list():
    l = typed.List()
    l.append(Order('', 0, 0.0, 0.0, '', '', 0, '', ''))
    l.clear()
    return l


@jitclass([
    ("long", types.ListType(typeof(Order('', 0, 0.0, 0.0, '', '', 0, '', '')))),
    ("short", types.ListType(typeof(Order('', 0, 0.0, 0.0, '', '', 0, '', ''))))
])
class OrderList:
    """
    A class representing open orders. Distinguishes between long and short orders.
    """

    def __init__(self):
        """
        Initializes the two lists as empty typed lists.
        """
        self.long = empty_order_list()
        self.short = empty_order_list()

    def add_long(self, orders: typed.List):
        """
        Adds orders to the long list.
        Requires a typed numba list of the form numba.typed.List().
        :param orders: Typed numba list of orders.
        :return:
        """
        for o in orders:
            self.long.append(o)

    def add_short(self, orders: typed.List):
        """
        Adds orders to the short list.
        Requires a typed numba list of the form numba.typed.List().
        :param orders: Typed numba list of orders.
        :return:
        """
        for o in orders:
            self.short.append(o)

    def delete_long(self, orders: typed.List):
        """
        Deletes the provided orders from the list of long orders.
        Requires a typed numba list of the form numba.typed.List().
        :param orders: Typed numba list of orders.
        :return:
        """
        to_delete = []
        for i in range(len(self.long)):
            for o in orders:
                if o.order_id == self.long[i].order_id:
                    to_delete.append(i)
                    break

        for i in sorted(to_delete, reverse=True):
            self.long.pop(i)

    def delete_short(self, orders: typed.List):
        """
        Deletes the provided orders from the list of short orders.
        Requires a typed numba list of the form numba.typed.List().
        :param orders: Typed numba list of orders.
        :return:
        """
        to_delete = []
        for i in range(len(self.short)):
            for o in orders:
                if o.order_id == self.short[i].order_id:
                    to_delete.append(i)
                    break

        for i in sorted(to_delete, reverse=True):
            self.short.pop(i)
