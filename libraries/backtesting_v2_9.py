# Developed by Raul Ikeda

import pandas as pd
import sys
import math
from copy import deepcopy
from datetime import datetime

import numpy as np


def sign(number):
    if number > 0:
        return 1
    elif number < 0:
        return -1
    return 0


class Event:

    BID, ASK, TRADE, CANDLE = ["BID", "ASK", "TRADE", "CANDLE"]

    def __init__(self, instrument, timestamp, type, price, quantity):
        self.instrument = instrument
        self.timestamp = timestamp
        self.type = type
        self.price = price
        self.quantity = quantity


class Order:

    id = 0

    NEW, PARTIAL, FILLED, REJECTED, CANCELED = [
        "NEW",
        "PARTIAL",
        "FILLED",
        "REJECTED",
        "CANCELED",
    ]

    B, S, SS = ["BUY", "SELL", "SELL SHORT"]

    @staticmethod
    def nextId():
        Order.id += 1
        return Order.id

    def __init__(self, instrument, side, quantity, price):
        self.id = Order.nextId()
        self.owner = 0
        self.instrument = instrument
        self.status = Order.NEW
        self.timestamp = ""
        if side not in [Order.B, Order.S, Order.SS]:
            raise Exception("Invalid order side")
        self.side = side
        self.quantity = quantity
        self.price = price
        self.executed = 0
        self.average = 0

    def print(self):
        return "{0} - {1} - {5}: {2}/{3}@{4}".format(
            self.id,
            self.timestamp,
            self.executed,
            self.quantity,
            self.price,
            self.instrument,
        )


class MarketData:

    TICK, HIST, INTR = ["TICK", "HIST", "INTR"]

    def __init__(self):

        self.events = {}

    def loadBBGTick(self, file, instrument):

        with open(file, "r") as file:
            data = file.read()

        events = data.split("\n")
        events = events[1:]
        for event in events:
            cols = event.split(";")
            if len(cols) == 4:
                date = datetime.strptime(cols[0], "%d/%m/%Y %H:%M:%S")
                price = float(cols[2].replace(",", "."))
                quantity = int(cols[3])
                type = cols[1]

                if date.toordinal() not in self.events:
                    self.events[date.toordinal()] = []

                self.events[date.toordinal()].append(
                    Event(instrument, date, type, price, quantity)
                )

    def loadYAHOOHist(self, file, instrument, type=Event.CANDLE):

        with open(file, "r") as file:
            data = file.read()

        events = data.split("\n")
        events = events[1:]
        for event in events:
            cols = event.split(",")
            if len(cols) == 7 and cols[1] != "null":

                date = datetime.strptime(cols[0], "%Y-%m-%d")
                price = (float(cols[1]), float(cols[2]),
                         float(cols[3]), float(cols[5]))
                quantity = float(cols[6])
                # quantity = 0

                if date.toordinal() not in self.events:
                    self.events[date.toordinal()] = []

                self.events[date.toordinal()].append(
                    Event(instrument, date, type, price, quantity)
                )

    def loadBBGIntr(self, file, instrument, type=Event.CANDLE):

        with open(file, "r") as file:
            data = file.read()

        events = data.split("\n")
        events = events[1:]
        for event in events:
            cols = event.split(";")
            if len(cols) == 5:

                date = datetime.strptime(cols[0], "%d/%m/%Y %H:%M:%S")
                price = (
                    float(cols[1].replace(",", ".")),
                    float(cols[3].replace(",", ".")),
                    float(cols[4].replace(",", ".")),
                    float(cols[2].replace(",", ".")),
                )
                quantity = 0

                if date.timestamp() not in self.events:
                    self.events[date.timestamp()] = []

                self.events[date.timestamp()].append(
                    Event(instrument, date, type, price, quantity)
                )

    def run(self, ts):
        dates = list(self.events.keys())
        dates.sort()
        for date in dates:
            for event in self.events[date]:
                ts.inject(event)


class Trade:

    def __init__(self):
        self.timestamp = ""
        self.position = {}
        self.orders = []
        self.fee = 0
        self.tax = 0
        self.avg_sell_price = 0
        self.avg_buy_price = 0
        self.sell_flow = 0
        self.buy_flow = 0
        self.max_alloc = 0
        self.ret = 0
        self.net_ret = 0
        self.max_profit_high = 0
        self.max_profit_close = 0
        self.max_dd_low = 0
        self.max_dd_close = 0
        self.events = []

    def zeroed(self):
        for pos in self.position.values():
            if pos != 0:
                return False
        return True

    def update_alloc(self):
        alloc = self.sell_flow + self.buy_flow
        self.max_alloc = max(max(self.max_alloc, alloc), -alloc)

    def partial_result(self, prices, leverage):
        result = self.sell_flow + self.buy_flow
        for instrument in self.position:
            result += self.position[instrument] * \
                prices[instrument][3] * leverage
        return result

    def partial_result_high(self, prices, leverage):
        result = self.sell_flow + self.buy_flow
        for instrument in self.position:
            result += self.position[instrument] * \
                prices[instrument][1] * leverage
        return result

    def partial_result_low(self, prices, leverage):
        result = self.sell_flow + self.buy_flow
        for instrument in self.position:
            result += self.position[instrument] * \
                prices[instrument][2] * leverage
        return result


class Strategy:

    unid = 0

    @staticmethod
    def next_id():
        Strategy.unid += 1
        return Strategy.unid

    def __new__(cls, *args, **kwargs):
        instance = super(Strategy, cls).__new__(cls)

        instance.id = Strategy.next_id()  # make sure not to override it

        instance._prices = {}  # vector of prices by instrument
        instance._last = {}  # dictionary of last prices by instrument

        instance._trade = Trade()  # Actual open trade
        instance._trades = []  # list of all trades
        instance._days = {}  # Struct of daily aggregated mectrics

        instance._orders = {}  # List of orders for flow control

        # Fee, Tax, Carry and Capital information
        instance.fee_order = 0.1  # per order
        instance.fee_flow = 0.00  # 0%
        instance.tax_flow_buy = 0
        instance.tax_flow_sell = 0.00001  # 0.001% Tax
        instance.tax_profit = 0.14999  # 15% Tax with paid flow sell
        instance.init_capital = 100000
        instance.avail_capital = instance.init_capital
        instance.risk_free_rate = 13.75

        # ToDo: Move it probably on instrument level
        instance.margin = 1  # 100% = cash
        instance.leverage = 1  # leverage multiplier

        return instance

    def __init__(self):
        pass

    # Functions for users to override, there should be a _[function] to take care data before calling them:

    def cancel(self, owner, id):
        pass

    def submit(self, id, orders):
        pass

    def receive(self, event):
        pass

    def fill(self, id, timestamp, instrument, price, quantity, status):
        pass

    def inject(self, event):
        pass

    # Internal functions:

    def _receive(self, event):

        # Save last price in dictionary
        if event.type == Event.CANDLE:
            self._last[event.instrument] = event.price
        elif event.type == Event.TRADE:
            self._last[event.instrument] = [
                event.price,
                event.price,
                event.price,
                event.price,
            ]

        # Save the prices in a vector
        if event.instrument not in self._prices:
            self._prices[event.instrument] = []
        self._prices[event.instrument].append(event.price)

        # New implementation Results:

        if (
            not self._trade.zeroed() and event.instrument in self._last
        ):  # new price event and strategy has allocation

            # high/low: consider the high price of the event/day, this could better/worst case scenario
            self._trade.max_profit_high = max(
                self._trade.max_profit_high,
                self._trade.partial_result_low(self._last, self.leverage),
            )
            self._trade.max_profit_high = max(
                self._trade.max_profit_high,
                self._trade.partial_result_high(self._last, self.leverage),
            )
            self._trade.max_dd_low = min(
                self._trade.max_dd_low,
                self._trade.partial_result_high(self._last, self.leverage),
            )
            self._trade.max_dd_low = min(
                self._trade.max_dd_low,
                self._trade.partial_result_low(self._last, self.leverage),
            )

            # Classic max Profit/Drawdown
            self._trade.max_profit_close = max(
                self._trade.max_profit_close,
                self._trade.partial_result(self._last, self.leverage),
            )
            self._trade.max_dd_close = min(
                self._trade.max_dd_close,
                self._trade.partial_result(self._last, self.leverage),
            )

            # Number of events each trade
            if event.timestamp not in self._trade.events:
                self._trade.events.append(event.timestamp)

        # This is not trivial. A strategy can be intraday and it might count Sharpe and Carry daily
        if event.timestamp not in self._days:  # check if it is a new day

            if len(self._days) > 0:  # if this is not the first day of the simulation
                # this data is from one day before the actual timestamp (after day ends)

                # Remember: can be multi-instrument or intraday
                max_timestamp = max(self._days.keys())

                # first index is the gross result
                self._days[max_timestamp][0] = (
                    self.avail_capital
                    + self._trade.partial_result(self._last, self.leverage)
                )
                daily_rate = (1 + self.risk_free_rate / 100) ** (1 / 252) - 1
                # second index is the carry revenue
                carry = (self.avail_capital -
                         self._trade.max_alloc) * daily_rate
                self._days[max_timestamp][1] = carry

                self._days[max_timestamp][2] = self._trade.max_dd_close

                # compound rates considering daily rate
                # it is the revenue over unused balance
                self.avail_capital += carry

            # this is filled as fresh new day, temp values
            self._days[event.timestamp] = [self.avail_capital, 0, 0]

        # Send event to user
        self.receive(event)

    # Event of filled order. It happens before yielding a fill to child strategy implementation
    def _fill(self, id, timestamp, instrument, price, quantity, status):

        if price != 0:  # some fill events are just status update

            # New Results implementation:

            if quantity != 0:

                # Update position quantity on instrument
                if instrument not in self._trade.position:
                    self._trade.position[instrument] = 0

                self._trade.position[instrument] += quantity

                # order list of a trade. One trade may contain n orders
                if id not in self._trade.orders:
                    self._trade.orders.append(
                        [id, timestamp, instrument, price, quantity]
                    )
                    self._trade.fee += self.fee_order

                # If it was a BUY
                if quantity > 0:

                    self._trade.buy_flow -= (
                        price * quantity * self.leverage
                    )  # cash flow

                    self._trade.fee += self.fee_flow * quantity * price
                    self._trade.tax -= self.tax_flow_buy * quantity * price
                else:

                    self._trade.sell_flow -= (
                        price * quantity * self.leverage
                    )  # cash flow

                    self._trade.fee -= self.fee_flow * quantity * price
                    self._trade.tax -= self.tax_flow_sell * quantity * price

                # Update max alloc to calculate return
                self._trade.update_alloc()

                # If the trade is completed zeroed, the trade is over
                if self._trade.zeroed():

                    # Update P&L (sum of cashflows)
                    self._trade.pnl = self._trade.sell_flow + self._trade.buy_flow

                    # Revenue tax
                    if self._trade.pnl > 0:
                        self._trade.tax += self.tax_profit * self._trade.pnl

                    # Update capital, this is not the result, it is balance
                    self.avail_capital += (
                        self._trade.pnl - self._trade.tax - self._trade.fee
                    )
                    # Return! Think on multi-instrument or arbitrage strategy to understand it
                    self._trade.ret = self._trade.pnl / self._trade.max_alloc
                    # Net return - including fees and taxes
                    self._trade.net_ret = (
                        self._trade.pnl - self._trade.fee - self._trade.tax
                    ) / self._trade.max_alloc

                    # Archive the trade and start another one
                    self._trades.append(self._trade)

                    # It is always available even though it is not used
                    self._trade = Trade()

            # Event to users only if price is not zero:
            self.fill(id, timestamp, instrument, price, quantity, status)

    def close(self):  # close all open positions

        for instrument, position in self._trade.position.items():
            if position > 0:
                self.submit(self.id, Order(instrument, Order.S, position, 0))
            elif position < 0:
                self.submit(self.id, Order(instrument, Order.B, position, 0))

        # Fill last item in vector days
        max_timestamp = max(self._days.keys())
        self._days[max_timestamp][0] = self.avail_capital + self._trade.partial_result(
            self._last, self.leverage
        )
        daily_rate = (1 + self.risk_free_rate / 100) ** (1 / 252) - 1
        self._days[max_timestamp][1] = (
            self.avail_capital - self._trade.max_alloc
        ) * daily_rate

    def summary(self):

        if len(self._trades) == 0:
            res = "No trades in the period\n\n"
            gross = 0
            fee = 0
            tax = 0
        else:

            res = ""
            res += "Gross Profit: ${0:.2f}\n".format(
                sum([trade.pnl for trade in self._trades if trade.pnl > 0])
            )
            res += "Gross Loss: ${0:.2f}\n".format(
                sum([trade.pnl for trade in self._trades if trade.pnl < 0])
            )
            res += "Gross Total: ${0:.2f}\n\n".format(
                sum([trade.pnl for trade in self._trades])
            )

            res += "Number of trades: {0}\n".format(len(self._trades))
            res += "Hitting Ratio: {0:.2f}%\n".format(
                100
                * len([trade.pnl for trade in self._trades if trade.pnl > 0])
                / len(self._trades)
            )
            res += "Number of profit trades: {0}\n".format(
                len([trade.pnl for trade in self._trades if trade.pnl > 0])
            )
            res += "Number of loss trades: {0}\n".format(
                len([trade.pnl for trade in self._trades if trade.pnl < 0])
            )
            res += "Average number of events per trade: {0:.2f}\n\n".format(
                np.mean([len(trade.events) for trade in self._trades])
            )

            win = [trade.pnl for trade in self._trades if trade.pnl > 0]
            loss = [trade.pnl for trade in self._trades if trade.pnl < 0]

            if len(win) > 0:
                res += "Max win trade: ${0:.2f}\n".format(max(win))
                res += "Avg win trade: ${0:.2f}\n".format(np.mean(win))
            else:
                res += "Max win trade: $-\n"
                res += "Avg win trade: $-\n"

            if len(loss) > 0:
                res += "Max loss trade: ${0:.2f}\n".format(min(loss))
                res += "Avg loss trade: ${0:.2f}\n".format(np.mean(loss))
            else:
                res += "Max loss trade: $-\n"
                res += "Avg loss trade: $-\n"

            res += "Avg all trades: ${0:.2f}\n".format(
                np.mean([trade.pnl for trade in self._trades])
            )
            if len(win) > 0 and len(loss) > 0:
                res += "Win/Loss ratio: {0:.2f}\n\n".format(
                    -np.mean(win) / np.mean(loss)
                )
            else:
                res += "Win/Loss ratio: -\n\n"

            res += "Max Profit: ${0:.2f}\n".format(
                max([trade.max_profit_close for trade in self._trades])
            )
            res += "Max Profit High/Low: ${0:.2f}\n".format(
                max([trade.max_profit_high for trade in self._trades])
            )
            res += "Max Drawdown: ${0:.2f}\n".format(
                min([trade.max_dd_close for trade in self._trades])
            )
            res += "Max Drawdown High/Low: ${0:.2f}\n\n".format(
                min([trade.max_dd_low for trade in self._trades])
            )

            max_alloc = max([trade.max_alloc for trade in self._trades])
            res += "Max Allocation: ${0:.2f}\n".format(max_alloc)
            res += "Avg Allocation: ${0:.2f}\n".format(
                np.mean([trade.max_alloc for trade in self._trades])
            )
            res += "Max Cash Required (margin): ${0:.2f}\n\n".format(
                max_alloc * self.margin
            )

            gross = sum([trade.pnl for trade in self._trades])
            fee = sum([trade.fee for trade in self._trades])
            tax = sum([trade.tax for trade in self._trades])

            res += "Gross Total: ${0:.2f}\n".format(gross)
            res += "Total Fees: ${0:.2f}\n".format(fee)
            res += "Total Taxes: ${0:.2f}\n".format(tax)
            res += "Net Total: ${0:.2f}\n\n".format(gross - fee - tax)

            res += "Gross Return: {0:.2f}%\n".format(
                100 * sum([trade.ret for trade in self._trades])
            )
            res += "Average Return: {0:.2f}%\n".format(
                100 * np.mean([trade.ret for trade in self._trades])
            )
            res += "Net Return: {0:.2f}%\n".format(
                100 * sum([trade.net_ret for trade in self._trades])
            )
            res += "Net Return Avg Alocation: {0:.2f}%\n\n".format(
                100
                * (gross - fee - tax)
                / np.mean([trade.max_alloc for trade in self._trades])
            )

        res += "Number of days: {}\n".format(len(self._days))
        res += "Initial Capital: ${0:.2f}\n".format(self.init_capital)
        daily_rate = (1 + self.risk_free_rate / 100) ** (1 / 252) - 1
        res += "Risk Free Rate: {0:.2f}% yearly/{1:.4f}% daily\n".format(
            self.risk_free_rate, 100 * daily_rate
        )
        carry = sum([day[1] for day in self._days.values()])
        res += "Total Carry: ${0:.2f}\n".format(carry)
        res += "Net Total + Carry: ${0:.2f}\n".format(
            gross - fee - tax + carry)
        ret_cap = (gross - fee - tax + carry) / self.init_capital
        res += "Net Return Capital: {0:.2f}%\n".format(100 * ret_cap)
        res += "Net Return Capital Yearly: {0:.2f}%\n\n".format(
            100 * ((1 + ret_cap) ** (252 / len(self._days)) - 1)
        )

        return res


# Class that deals with order book
class Book:

    def __init__(self, instrument, fill):

        self.instrument = instrument
        self.fill = fill

        # Market data
        self.bid = None  # bid queue
        self.ask = None  # ask queue
        self.trade = None  # trades list
        self.timestamp = None  # last update

        # Pending Orders:
        self.orders = []

    def inject(self, event):  # received new event

        if event.instrument == self.instrument:
            self.timestamp = event.timestamp

            # If it is a candle, need to save the HLC
            if event.type == Event.CANDLE:
                high = event.price[1]
                low = event.price[2]
                event.price = event.price[3]

            else:  # consider HL as close price for convenience
                high = event.price
                low = event.price

            if event.type == Event.BID or event.type == Event.CANDLE:
                # if BID or candle, update book
                self.bid = event
                # try to match the pending orders
                for order in self.orders:
                    if order.quantity < 0:
                        # if high is higher, it would match at order price
                        if order.price <= high:
                            rem = order.quantity - order.executed

                            if event.quantity == 0:
                                qty = rem
                            else:
                                qty = max(rem, -event.quantity)

                            average = order.average * order.executed + qty * order.price

                            order.executed += qty
                            order.average = average / order.executed

                            if order.quantity == order.executed:
                                order.status = Order.FILLED

                            self.fill(
                                order.id, self.timestamp, order.price, qty, order.status
                            )

            # Same logic as BID
            if event.type == Event.ASK or event.type == Event.CANDLE:
                self.ask = event
                for order in self.orders:
                    if order.quantity > 0:
                        if order.price >= low:
                            rem = order.quantity - order.executed

                            if event.quantity == 0:
                                qty = rem
                            else:
                                qty = min(rem, event.quantity)

                            average = order.average * order.executed + qty * order.price

                            order.executed += qty
                            order.average = average / order.executed

                            if order.quantity == order.executed:
                                order.status = Order.FILLED

                            self.fill(
                                order.id, self.timestamp, order.price, qty, order.status
                            )

            # This is for tick-by-tick data, consider the pending orders are in first position
            if event.type == Event.TRADE:
                self.trade = event
                for order in self.orders:
                    if order.quantity > 0 and order.price >= event.price:
                        rem = order.quantity - order.executed

                        if event.quantity == 0:
                            qty = rem
                        else:
                            qty = min(rem, event.quantity)

                        average = order.average * order.executed + qty * order.price

                        order.executed += qty
                        order.average = average / order.executed

                        if order.quantity == order.executed:
                            order.status = Order.FILLED

                        self.fill(
                            order.id, event.timestamp, order.price, qty, order.status
                        )

                    if order.quantity < 0 and order.price <= event.price:
                        rem = order.quantity - order.executed

                        if event.quantity == 0:
                            qty = rem
                        else:
                            qty = max(rem, -event.quantity)

                        average = order.average * order.executed + qty * order.price

                        order.executed += qty
                        order.average = average / order.executed

                        if order.quantity == order.executed:
                            order.status = Order.FILLED

                        self.fill(
                            order.id, event.timestamp, order.price, qty, order.status
                        )

            # Handle filled orders
            i = 0
            while i < len(self.orders):
                if self.orders[i].status == Order.FILLED:
                    del self.orders[i]
                else:
                    i += 1

    def submit(self, order):  # received new order

        if order is not None:  # avoid error

            # correct quantity according to side declaration
            if (order.side == Order.S or order.side == Order.SS) and order.quantity > 0:
                order.quantity = -order.quantity
            if order.side == Order.B and order.quantity < 0:
                order.quantity = -order.quantity

            # MKT order - at any price, fill with book
            if order.price == 0:
                if order.quantity > 0:  # BUY order
                    if self.ask.quantity == 0:  # there is no real ask in book
                        order.executed = order.quantity
                    else:  # sometimes can't fill everything and the rest remain pending
                        order.executed = min(
                            [self.ask.quantity, order.quantity])

                    order.average = self.ask.price
                    order.status = Order.FILLED

                    # yield the fill event
                    self.fill(
                        order.id,
                        self.timestamp,
                        order.average,
                        order.executed,
                        order.status,
                    )

                elif order.quantity < 0:  # same logic as BUY order
                    if self.bid.quantity == 0:
                        order.executed = order.quantity
                    else:
                        order.executed = max(
                            [-self.bid.quantity, order.quantity])

                    order.average = self.bid.price
                    order.status = Order.FILLED

                    self.fill(
                        order.id,
                        self.timestamp,
                        order.average,
                        order.executed,
                        order.status,
                    )

            else:  # LMT order - fill only if price is better than in order

                # If BUY and there is ask in book and price is equal or better
                if (
                    self.ask is not None
                    and order.quantity > 0
                    and order.price >= self.ask.price
                ):
                    if self.ask.quantity == 0:
                        order.executed = order.quantity
                        order.average = self.ask.price  # fill with book price
                        order.status = Order.FILLED
                    else:  # can't fill everything
                        order.executed = min(
                            [self.ask.quantity, order.quantity])
                        order.average = self.ask.price  # fill with book price
                        if order.executed == order.quantity:
                            order.status = Order.FILLED
                        else:
                            order.status = Order.PARTIAL
                            self.orders.append(order)
                    self.fill(
                        order.id,
                        self.timestamp,
                        order.average,
                        order.executed,
                        order.status,
                    )
                # Same logic as BUY, but for SELL
                elif (
                    self.bid is not None
                    and order.quantity < 0
                    and order.price <= self.bid.price
                ):
                    if self.bid.quantity == 0:
                        order.executed = order.quantity
                        order.average = self.bid.price
                        order.status = Order.FILLED
                    else:
                        order.executed = max(
                            [-self.bid.quantity, order.quantity])
                        order.average = self.bid.price
                        if order.executed == order.quantity:
                            order.status = Order.FILLED
                        else:
                            order.status = Order.PARTIAL
                            self.orders.append(order)
                    self.fill(
                        order.id,
                        self.timestamp,
                        order.average,
                        order.executed,
                        order.status,
                    )
                elif order.quantity != 0:
                    self.orders.append(order)

    def cancel(self, id):  # Cancel an order
        i = 0
        while i < len(self.orders):
            if self.orders[i].id == id:
                order = self.orders[i]
                del self.orders[i]
                order.status = Order.CANCELED
                # Yield cancel event as update status with no quantity or price
                self.fill(order.id, self.timestamp, 0, 0, order.status)
                i = len(self.orders)
            else:
                i += 1


class TradingSystem:

    def __init__(self):
        self.books = {}
        self.position = {}
        self.orders = {}
        self.listeners = {}
        self.strategies = {}

    def createBook(self, instrument):
        if instrument not in self.books:
            self.books[instrument] = Book(instrument, self.fill)

        if instrument not in self.position:
            self.position[instrument] = {}

        if instrument not in self.listeners:
            self.listeners[instrument] = []

    def inject(self, event):
        instrument = event.instrument
        if instrument in self.books:
            self.books[instrument].inject(deepcopy(event))

            for id in self.listeners[instrument]:
                if id in self.strategies:
                    self.strategies[id]._receive(event)

    def subscribe(self, instrument, strategy):
        if strategy.id not in self.strategies:
            self.strategies[strategy.id] = strategy
            strategy.cancel = self.cancel
            strategy.submit = self.submit
            strategy.inject = self.inject

        if instrument in self.books:
            if strategy.id not in self.position[instrument]:
                self.position[instrument][strategy.id] = 0

            if strategy.id not in self.listeners[instrument]:
                self.listeners[instrument].append(strategy.id)

    def submit(self, id, order):

        if order is not None:

            order.owner = id
            instrument = order.instrument

            if instrument in self.position:
                if id in self.position[instrument]:
                    position = self.position[instrument][id]

            if sign(position) * sign(position + order.quantity) == -1:
                order.status = Order.REJECTED
                if id in self.strategies:
                    strategy = self.strategies[id]
                    strategy._fill(
                        order.id, order.timestamp, instrument, 0, 0, order.status
                    )
            else:
                if order.id not in self.orders:
                    self.orders[order.id] = order

                if instrument in self.books:
                    self.books[instrument].submit(order)

    def cancel(self, owner, id):
        if id in self.orders:
            if self.orders[id].owner == owner:
                instrument = self.orders[id].instrument
                if instrument in self.books:
                    self.books[instrument].cancel(id)

    def fill(self, id, timestamp, price, quantity, status):

        if id in self.orders:

            order = self.orders[id]
            instrument = order.instrument
            owner = order.owner

            if instrument in self.position:
                if owner in self.position[instrument]:
                    self.position[instrument][owner] += quantity

            if owner in self.strategies:
                strategy = self.strategies[owner]
                strategy._fill(id, timestamp, instrument,
                               price, quantity, status)


# Main Functions


def evaluate(strategy, type, files):
    data = MarketData()

    ts = TradingSystem()

    for instrument, file in files.items():
        ts.createBook(instrument)
        ts.subscribe(instrument, strategy)
        if file != "":
            if type == MarketData.TICK:
                data.loadBBGTick(file, instrument)
            elif type == MarketData.HIST:
                data.loadYAHOOHist(file, instrument)
            elif type == MarketData.INTR:
                data.loadBBGIntr(file, instrument)

    data.run(ts)
    strategy.close()
    return strategy.summary()


def evaluateTick(strategy, files):
    return evaluate(strategy, MarketData.TICK, files)


def evaluateHist(strategy, files):
    return evaluate(strategy, MarketData.HIST, files)


def evaluateIntr(strategy, files):
    return evaluate(strategy, MarketData.INTR, files)


class SimpleStrategy(Strategy):

    def __init__(self, func, capital=100000, instruments=[], fill=None):
        self.func = func

        self.instruments = instruments

        if len(instruments) <= 1:
            self.last_price = 0
            self.position = 0
            self.prices = pd.DataFrame(
                {"Open": [], "High": [], "Low": [], "Close": [], "Volume": []})
        else:
            self.last_price = {}
            self.position = {}
            self.prices = {}

            for inst in instruments:
                self.last_price[inst] = 0
                self.position[inst] = 0
                self.prices[inst] = pd.DataFrame(
                    {"Open": [], "High": [], "Low": [], "Close": [], "Volume": []})

        self.init_capital = capital
        self.avail_capital = capital
        self.capital = capital
        self.first = True
        self.filled = False
        self.main_module = sys.modules["__main__"]
        self.print_fill = False

        self.extra_fill = None
        if fill is not None:
            self.extra_fill = fill

    def receive(self, event):

        if event.type == Event.CANDLE:

            if len(self.instruments) <= 1:
                self.last_price = event.price[3]
                self.prices.loc[event.timestamp] = list(
                    event.price) + [event.quantity]
            elif event.instrument in self.last_price:
                self.last_price[event.instrument] = event.price[3]
                self.prices[event.instrument].loc[event.timestamp] = list(
                    event.price) + [event.quantity]

            setattr(self.main_module, "first_event", self.first)
            setattr(self.main_module, "filled", self.filled)
            setattr(self.main_module, "last_price", self.last_price)
            setattr(self.main_module, "prices", self.prices)

            try:
                self.func()
            except:
                pass

            self.first = False
            self.filled = False

    def fill(self, id, timestamp, instrument, price, quantity, status):

        if self.print_fill:
            print("{} {} order filled: {}@{}".format(timestamp,
                  instrument, quantity, price))

        if len(self.instruments) <= 1:
            self.position += quantity
        else:
            self.position[instrument] += quantity

        self.capital -= price * quantity
        setattr(self.main_module, "position", self.position)
        setattr(self.main_module, "capital", self.capital)

        if id in self._orders:
            if self._orders[id].quantity == self._orders[id].executed:
                self.filled = True
                del self._orders[id]

        if self.extra_fill is not None:
            self.extra_fill(id, timestamp, instrument, price, quantity, status)

    def inject(self, event):
        pass


# Global variable:


# Update the caller's globals
main_module = sys.modules["__main__"]


prices = pd.DataFrame(
    {"Open": [], "High": [], "Low": [], "Close": [], "Volume": []})
setattr(main_module, "prices", prices)

position = 0
setattr(main_module, "position", position)
capital = 100000
setattr(main_module, "capital", capital)

first_event = True
setattr(main_module, "first_event", first_event)
last_price = 0
setattr(main_module, "last_price", last_price)
filled = False
setattr(main_module, "filled", filled)


def dummy():
    pass


instrument = ""
setattr(main_module, "instrument", instrument)

strategy = [
    SimpleStrategy(lambda: print("Hello World!")),
    Strategy(),
]


def CANCEL_ALL():

    for order in strategy[0]._orders:
        strategy[0].cancel(strategy[0].id, order)

    strategy[0]._orders = {}


def BUY(price=0, quantity=0, symbol=''):
    _quantity = quantity
    instrument = getattr(main_module, "instrument", "")

    if symbol == '':
        symbol = instrument[0]

    if quantity == 0 and strategy[0].position == 0:
        _quantity = strategy[0].capital // strategy[0].last_price

    order = Order(symbol, Order.B, _quantity, price)
    strategy[0]._orders[order.id] = order
    strategy[0].submit(strategy[0].id, order)


def SELL(price=0, quantity=0, symbol=''):
    _quantity = min(quantity, strategy[0].position)
    instrument = getattr(main_module, "instrument", "")

    if symbol == '':
        symbol = instrument[0]

    if quantity == 0:
        _quantity = strategy[0].position

    if _quantity > 0:
        order = Order(symbol, Order.S, _quantity, price)
        strategy[0]._orders[order.id] = order
        strategy[0].submit(strategy[0].id, order)


def SHORT(price=0, quantity=0, symbol=''):
    _quantity = quantity
    instrument = getattr(main_module, "instrument", "")

    if symbol == '':
        symbol = instrument[0]

    if quantity == 0 and strategy[0].position == 0:
        _quantity = strategy[0].capital // strategy[0].last_price

    order = Order(symbol, Order.SS, _quantity, price)
    strategy[0]._orders[order.id] = order
    strategy[0].submit(strategy[0].id, order)


def COVER(price=0, quantity=0, symbol=''):
    _quantity = max(quantity, strategy[0].position)
    instrument = getattr(main_module, "instrument", "")

    if symbol == '':
        symbol = instrument[0]

    if quantity == 0:
        _quantity = strategy[0].position

    if _quantity < 0:
        order = Order(symbol, Order.B, -_quantity, price)
        strategy[0]._orders[order.id] = order
        strategy[0].submit(strategy[0].id, order)


def SEND(order):
    strategy[0].submit(strategy[0].id, order)


def simple_test(
    ticker,
    date_from,
    date_to,
    func,
    capital=1000000,
    risk_free_rate=11.25,
    print_fill=False,
    files={},
    file_type=MarketData.INTR
):

    main_module = sys.modules["__main__"]

    instrument = []

    if hasattr(ticker, '__iter__'):
        prices = {}
        position = {}
        for element in ticker:
            instrument.append(element)
            prices[element] = pd.DataFrame(
                {"Open": [], "High": [], "Low": [], "Close": [], "Volume": []})
            position[element] = 0

        setattr(main_module, "prices", prices)

    else:
        instrument = [ticker]

    setattr(main_module, "instrument", instrument)

    setattr(main_module, "position", position)
    setattr(main_module, "capital", capital)

    strategy[0] = SimpleStrategy(func, capital, instrument)
    strategy[1] = Strategy()

    for i in range(len(strategy)):

        strategy[i].init_capital = capital
        strategy[i].avail_capital = capital
        strategy[i].capital = capital
        strategy[i].risk_free_rate = risk_free_rate

    setattr(main_module, "first_event", strategy[0].first)
    setattr(main_module, "last_price", strategy[0].last_price)
    setattr(main_module, "filled", strategy[0].filled)
    strategy[0].print_fill = print_fill

    # strategy = SimpleStrategy(func, capital)

    # prices.drop(prices.index, inplace=True)

    mkt_data = MarketData()

    ts = TradingSystem()

    for inst in instrument:

        ts.createBook(inst)
        for i in range(len(strategy)):
            ts.subscribe(inst, strategy[i])

    if len(files) == 0:

        import yfinance as yf

        data = yf.download(instrument, start=date_from,
                           end=date_to, progress=False)

        # Ler os dados do JSON
        df = data.dropna()

        # import yahoofinancials as yf

        # data = yf.YahooFinancials(instrument).get_historical_price_data(
        #    date_from, date_to, "daily"
        # )

        # Ler os dados do JSON
        # raw = pd.DataFrame(data[instrument]["prices"]).dropna()
        # Converter a data para o tipo correto datetime
        # raw["formatted_date"] = pd.to_datetime(raw["formatted_date"])
        # Indica a data como o índice de cada linha
        # raw = raw.set_index("formatted_date")
        # Removendo as colunas que não interessam
        # df = raw.iloc[:, 1:]
        # df = df.reindex(columns=["open", "high", "low", "close", "adjclose", "volume"])

        for index, row in df.iterrows():
            date = index
            mkt_data.events[date.toordinal()] = []
            for inst in instrument:
                factor = row["Adj Close"][inst] / row["Close"][inst]
                price = (
                    row["Open"][inst] * factor,
                    row["High"][inst] * factor,
                    row["Low"][inst] * factor,
                    row["Adj Close"][inst],
                )
                quantity = row["Volume"][inst]

                mkt_data.events[date.toordinal()].append(
                    Event(inst, date, Event.CANDLE, price, quantity))
    else:
        for inst, file in files.items():
            if file != "":
                if file_type == MarketData.TICK:
                    mkt_data.loadBBGTick(file, inst)
                elif file_type == MarketData.HIST:
                    mkt_data.loadYAHOOHist(file, inst)
                elif file_type == MarketData.INTR:
                    mkt_data.loadBBGIntr(file, inst)

    mkt_data.run(ts)
    strategy[0].close()
    strategy[1].close()
    return strategy[0].summary()
