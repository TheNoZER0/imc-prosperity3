from typing import Dict, List
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
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

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

    def run(self, state: TradingState) -> tuple[dict[Symbol,List[Order]], int, str]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        conversions = 1
        traderData = "SAMPLE"
        # Initialize the method output dict as an empty dict
        logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))
        result = {}

        # Iterate over all the keys (the available products) contained in the order dephts
        for product in state.order_depths.keys():

            # Check if the current product is the 'PEARLS' product, only then run the order logic
            if product == 'RAINFOREST_RESIN':
                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []
                
                # Get position
                MAX_POSITION = 20
                position = state.position.get(product, 0)
                
                # Initialize or retrieve trader data
                if not state.traderData:
                    trader_data = {
                        "resin_prices": [],
                        "bb_sma": None,
                        "bb_upper": None,
                        "bb_lower": None,
                        "last_signal": None  # To prevent repeated signals
                    }
                else:
                    try:
                        trader_data = json.loads(state.traderData)
                        if "resin_prices" not in trader_data:
                            trader_data["resin_prices"] = []
                            trader_data["bb_sma"] = None
                            trader_data["bb_upper"] = None
                            trader_data["bb_lower"] = None
                            trader_data["last_signal"] = None
                    except:
                        trader_data = {
                            "resin_prices": [],
                            "bb_sma": None,
                            "bb_upper": None,
                            "bb_lower": None,
                            "last_signal": None
                        }
                
                if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_bid = max(order_depth.buy_orders.keys())
                    current_price = (best_ask + best_bid) / 2
                    
                    # Update price history
                    trader_data["resin_prices"].append(current_price)
                    
                    # Keep only most recent 50 prices
                    if len(trader_data["resin_prices"]) > 50:
                        trader_data["resin_prices"] = trader_data["resin_prices"][-50:]
                    
                    # Fine-tuned Bollinger Bands strategy
                    if len(trader_data["resin_prices"]) >= 20:
                        window = trader_data["resin_prices"][-20:]
                        sma = sum(window) / 20
                        
                        # Calculate standard deviation
                        variance = sum((x - sma) ** 2 for x in window) / 20
                        std_dev = variance ** 0.5
                        
                        # Adjusted parameters - tighter bands for more signals
                        upper_band = sma + (1.5 * std_dev)  # Changed from 2.0 to 1.5
                        lower_band = sma - (1.5 * std_dev)  # Changed from 2.0 to 1.5
                        
                        # Store values
                        trader_data["bb_sma"] = sma
                        trader_data["bb_upper"] = upper_band
                        trader_data["bb_lower"] = lower_band
                        
                        # Dynamic position sizing - scale based on distance from bands
                        distance_from_band = 0
                        current_signal = None
                        if current_price <= lower_band:
                            current_signal = "BUY"
                            # Calculate how far below the band we are (as percentage)
                            distance_from_band = min(1.0, (lower_band - current_price) / std_dev)
                        elif current_price >= upper_band:
                            current_signal = "SELL"
                            # Calculate how far above the band we are (as percentage)
                            distance_from_band = min(1.0, (current_price - upper_band) / std_dev)
                        
                        # Only trade if signal is different from last time or None
                        if current_signal != trader_data["last_signal"]:
                            if current_signal == "BUY" and position < MAX_POSITION:
                                # Scale position size based on signal strength
                                signal_strength = 0.5 + (0.5 * distance_from_band)  # 50-100% of max position
                                max_buy = int(MAX_POSITION * signal_strength)
                                available_to_buy = max_buy - position
                                ask_volume = abs(order_depth.sell_orders[best_ask])
                                buy_qty = min(ask_volume, available_to_buy)
                                
                                if buy_qty > 0:
                                    orders.append(Order(product, best_ask, buy_qty))
                                    
                            elif current_signal == "SELL" and position > -MAX_POSITION:
                                # Scale position size based on signal strength
                                signal_strength = 0.5 + (0.5 * distance_from_band)  # 50-100% of max position
                                max_sell = int(MAX_POSITION * signal_strength)
                                available_to_sell = position + max_sell
                                bid_volume = abs(order_depth.buy_orders[best_bid])
                                sell_qty = min(bid_volume, available_to_sell)
                                
                                if sell_qty > 0:
                                    orders.append(Order(product, best_bid, -sell_qty))
                        
                        # Update the last signal
                        trader_data["last_signal"] = current_signal
                
                # Update trader data in state
                traderData = json.dumps(trader_data)
                result[product] = orders

            if product == 'KELP':
                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []

                # Gather the last 20 KELP trades (example uses state.market_trades)
                all_kelp_trades = state.market_trades.get(product, [])
                recent_prices = [t.price for t in all_kelp_trades[-20:]]

                MAX_POSITION = 20
                position = state.position.get(product, 0)

                # Fine-tuned range-based strategy
                if len(recent_prices) >= 20:
                    high = max(recent_prices)
                    low = min(recent_prices)
                    
                    # Adjusted threshold - more sensitive to range changes
                    threshold = (high - low) * 0.15  # Changed from 0.2 to 0.15
                    buy_level = low + threshold
                    sell_level = high - threshold
                    
                    # Trend detection - check if prices are making higher lows or lower highs
                    is_trending_up = False
                    is_trending_down = False
                    
                    # Look at recent price movement (last 5 prices)
                    if len(recent_prices) >= 5:
                        last_5 = recent_prices[-5:]
                        # Check if we're making higher lows (uptrend)
                        lows = [min(last_5[i:i+3]) for i in range(3)]
                        is_trending_up = lows[0] < lows[1] < lows[2]
                        # Check if we're making lower highs (downtrend)
                        highs = [max(last_5[i:i+3]) for i in range(3)]
                        is_trending_down = highs[0] > highs[1] > highs[2]
                    
                    # Only proceed if we have both buy and sell orders
                    if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
                        best_ask = min(order_depth.sell_orders.keys())
                        best_bid = max(order_depth.buy_orders.keys())
                        mid_price = (best_ask + best_bid) / 2
                        
                        # Buy if price is near the low AND we see signs of an uptrend
                        if mid_price < buy_level and position < MAX_POSITION and (is_trending_up or mid_price < low + (threshold * 0.5)):
                            ask_volume = abs(order_depth.sell_orders[best_ask])
                            buy_qty = min(ask_volume, MAX_POSITION - position)
                            if buy_qty > 0:
                                orders.append(Order(product, best_ask, buy_qty))
                        
                        # Sell if price is near the high AND we see signs of a downtrend
                        if mid_price > sell_level and position > -MAX_POSITION and (is_trending_down or mid_price > high - (threshold * 0.5)):
                            bid_volume = abs(order_depth.buy_orders[best_bid])
                            sell_qty = min(bid_volume, position + MAX_POSITION)
                            if sell_qty > 0:
                                orders.append(Order(product, best_bid, -sell_qty))

                # If fewer than 20 recent trades, you can default to your existing logic or do nothing
                else:
                    # Fallback: your existing LOW_BOUND / HIGH_BOUND logic
                    LOW_BOUND = 2013
                    HIGH_BOUND = 2027
                    if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
                        best_ask = min(order_depth.sell_orders.keys())
                        best_bid = max(order_depth.buy_orders.keys())
                        mid_price = (best_ask + best_bid) / 2
                        ask_volume = abs(order_depth.sell_orders[best_ask])
                        bid_volume = abs(order_depth.buy_orders[best_bid])

                        # Buy if near lower bound
                        if mid_price <= LOW_BOUND and position < MAX_POSITION:
                            buy_qty = min(ask_volume, MAX_POSITION - position)
                            if buy_qty > 0:
                                orders.append(Order(product, best_ask, buy_qty))

                        # Sell if near upper bound
                        if mid_price >= HIGH_BOUND and position > -MAX_POSITION:
                            sell_qty = min(bid_volume, position + MAX_POSITION)
                            if sell_qty > 0:
                                orders.append(Order(product, best_bid, -sell_qty))

                result[product] = orders
                
        logger.flush(state, result, conversions, traderData)     
        return result, conversions, traderData

