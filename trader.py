from typing import Dict, List
import numpy as np
import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()

class Trader:
    
    # position_limits = {
    #     'RAINFOREST_RESIN': 200,
    #     'KELP': 100,
    # }
    
    def get_best_bid(self, order_depth):
        """
        Returns the best bid price and its volume.
        If there are no bid orders, returns None.
        """
        if order_depth.buy_orders:
            best_bid_price = max(order_depth.buy_orders.keys())
            best_bid_volume = order_depth.buy_orders[best_bid_price]
            return best_bid_price, best_bid_volume
        else:
            return None, None

    def get_best_ask(self, order_depth):
        """
        Returns the best ask price and its volume.
        If there are no ask orders, returns None.
        """
        if order_depth.sell_orders:
            best_ask_price = min(order_depth.sell_orders.keys())
            best_ask_volume = order_depth.sell_orders[best_ask_price]
            return best_ask_price, best_ask_volume
        else:
            return None, None
    

    def mean_reversion_rainforesin(self, state, fair_value, threshold=6, base_order_qty=10):
        """
        Simplified mean reversion for RAINFOREST_RESIN:
        - Buys if the current mid price is below fair_value - threshold.
        - Sells if it is above fair_value + threshold.
        The function “walks” through the order book to accumulate a desired base order quantity.
        
        Parameters:
            state: TradingState with order_depths and position.
            fair_value: The assumed fair value (e.g., determined externally).
            threshold: Fixed threshold (default 2) for deviation from fair value.
            base_order_qty: The desired number of units to trade when a signal is triggered.
        
        Returns:
            List of Order objects to send.
        """
        orders = []
        product = "RAINFOREST_RESIN"
        order_depth = state.order_depths[product]

        # Sort asks: lowest price first; Sort bids: highest price first
        sorted_asks = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
        sorted_bids = sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)

        # Get the best ask and best bid directly from the order book
        best_ask, ask_vol = self.get_best_ask(order_depth)
        best_bid, bid_vol = self.get_best_bid(order_depth)

        # Retrieve current position; default to 0 if not available
        current_pos = state.position.get(product, 0)
        max_limit = 10  # maximum allowed absolute position

        # Allowed quantities for buying and selling based on the current position
        allowed_buy_qty = max(0, max_limit - current_pos)
        allowed_sell_qty = max(0, max_limit + current_pos)

        # --- Trading Logic ---
        # Buy signal: if the best ask is below (fair_value - threshold)
        if best_ask is not None and best_ask < fair_value + threshold:
            desired_qty = min(abs(current_pos) - base_order_qty, allowed_buy_qty)
            if desired_qty > 0:
                qty_remaining = desired_qty
                for price, vol in sorted_asks:
                    if price <= fair_value + threshold:
                        trade_qty = min(vol, qty_remaining)
                        orders.append(Order(product, price, trade_qty))  # positive quantity for buy
                        qty_remaining -= trade_qty
                        if qty_remaining <= 0:
                            break

        # Sell signal: if the best bid is above (fair_value + threshold)
        elif best_bid is not None and best_bid > fair_value - threshold:
            desired_qty = min(base_order_qty, allowed_sell_qty)
            if desired_qty > 0:
                qty_remaining = desired_qty
                for price, vol in sorted_bids:
                    if price >= fair_value - threshold:
                        trade_qty = min(vol, qty_remaining)
                        orders.append(Order(product, price, -trade_qty))  # negative quantity for sell
                        qty_remaining -= trade_qty
                        if qty_remaining <= 0:
                            break

        return orders


    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))
        result = {}

        # Iterate over all the keys (the available products) contained in the order dephts
        for product in state.order_depths.keys():

            # Check if the current product is the 'PEARLS' product, only then run the order logic
            # if product == 'KELP':

            #     # Retrieve the Order Depth containing all the market BUY and SELL orders for PEARLS
            #     order_depth: OrderDepth = state.order_depths[product]

            #     # Initialize the list of Orders to be sent as an empty list
            #     orders: list[Order] = []

            #     # Define a fair value for the PEARLS.
            #     # Note that this value of 1 is just a dummy value, you should likely change it!
            #     acceptable_price = 1

            #     # If statement checks if there are any SELL orders in the PEARLS market
            #     if len(order_depth.sell_orders) > 0:

            #         # Sort all the available sell orders by their price,
            #         # and select only the sell order with the lowest price
            #         best_ask = min(order_depth.sell_orders.keys())
            #         best_ask_volume = order_depth.sell_orders[best_ask]

            #         # Check if the lowest ask (sell order) is lower than the above defined fair value
            #         if best_ask < acceptable_price:

            #             # In case the lowest ask is lower than our fair value,
            #             # This presents an opportunity for us to buy cheaply
            #             # The code below therefore sends a BUY order at the price level of the ask,
            #             # with the same quantity
            #             # We expect this order to trade with the sell order
            #             print("BUY", str(-best_ask_volume) + "x", best_ask)
            #             orders.append(Order(product, best_ask, -best_ask_volume))

            #     # The below code block is similar to the one above,
            #     # the difference is that it find the highest bid (buy order)
            #     # If the price of the order is higher than the fair value
            #     # This is an opportunity to sell at a premium
            #     if len(order_depth.buy_orders) != 0:
            #         best_bid = max(order_depth.buy_orders.keys())
            #         best_bid_volume = order_depth.buy_orders[best_bid]
            #         if best_bid > acceptable_price:
            #             print("SELL", str(best_bid_volume) + "x", best_bid)
            #             orders.append(Order(product, best_bid, -best_bid_volume))

            #     # Add all the above the orders to the result dict
            #     result[product] = orders

            #     # Return the dict of orders
            #     # These possibly contain buy or sell orders for PEARLS
            #     # Depending on the logic above
            #     conversions = 1
            #     traderData = "SAMPLE"
                
            if product == 'RAINFOREST_RESIN':
                fair_value = 10000  
                resin_orders = self.mean_reversion_rainforesin(state, fair_value, threshold=1, base_order_qty=1)
                result["RAINFOREST_RESIN"] = resin_orders
                conversions = 1
                traderData = "RAINFOREST_RESIN"
                
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
    
    