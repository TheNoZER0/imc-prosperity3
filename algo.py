import json
import math
from typing import Any
import numpy as np
from datamodel import TradingState, Order, Symbol, Listing, Observation, OrderDepth, ProsperityEncoder, Trade

CROISSANTS = "CROISSANTS"
JAMS = "JAMS"
DJEMBES = "DJEMBES"
PICNIC_BASKET1 = "PICNIC_BASKET1"
PICNIC_BASKET2 = "PICNIC_BASKET2"

POSITION_LIMITS = {
    CROISSANTS: 250,
    JAMS: 350,
    DJEMBES: 60,
    PICNIC_BASKET1: 60,
    PICNIC_BASKET2: 100,
}

MIN_ARB_DIFF = 5
SCALING_FACTOR = 0.5
MAX_BASE_QTY = 10

BASE_TICK = 1
VOL_WINDOW = 20
VOL_SCALE = 0.1

mid_price_history = {sym: [] for sym in [CROISSANTS, JAMS, DJEMBES, PICNIC_BASKET1, PICNIC_BASKET2]}


def update_mid_price_history(state: TradingState, symbol: str) -> list[float]:
    mid = get_mid_price(state, symbol)
    hist = mid_price_history[symbol]
    hist.append(mid)
    if len(hist) > VOL_WINDOW:
        hist.pop(0)
    return hist


def estimate_volatility(prices: list[float]) -> float:
    return np.std(prices) if len(prices) >= 2 else 0


def get_mid_price(state: TradingState, symbol: str) -> float:
    od = state.order_depths[symbol]
    best_bid = max(od.buy_orders.keys()) if od.buy_orders else 0
    best_ask = min(od.sell_orders.keys()) if od.sell_orders else (best_bid + 1)
    return (best_bid + best_ask) / 2


def theoretical_basket1_price(croissant_price: float, jam_price: float, djembe_price: float) -> float:
    return 6 * croissant_price + 3 * jam_price + djembe_price


def theoretical_basket2_price(croissant_price: float, jam_price: float) -> float:
    return 4 * croissant_price + 2 * jam_price


def arbitrage_strategy(state: TradingState, base_qty: int = MAX_BASE_QTY) -> list[Order]:
    orders = []
    croissant_mid = get_mid_price(state, CROISSANTS)
    jam_mid = get_mid_price(state, JAMS)
    djembe_mid = get_mid_price(state, DJEMBES)

    basket1_theoretical = theoretical_basket1_price(croissant_mid, jam_mid, djembe_mid)
    basket2_theoretical = theoretical_basket2_price(croissant_mid, jam_mid)

    basket1_mid = get_mid_price(state, PICNIC_BASKET1)
    basket2_mid = get_mid_price(state, PICNIC_BASKET2)

    # Basket 1 Arbitrage
    diff1 = abs(basket1_theoretical - basket1_mid)
    if diff1 >= MIN_ARB_DIFF:
        qty1 = min(base_qty, POSITION_LIMITS[PICNIC_BASKET1] - state.position.get(PICNIC_BASKET1, 0))
        scale = SCALING_FACTOR * (diff1 / basket1_mid)
        adjusted_qty = max(1, int(qty1 * scale))
        if basket1_theoretical < basket1_mid:
            orders.append(Order(PICNIC_BASKET1, int(basket1_mid), -adjusted_qty))
            orders.append(Order(CROISSANTS, int(croissant_mid), 6 * adjusted_qty))
            orders.append(Order(JAMS, int(jam_mid), 3 * adjusted_qty))
            orders.append(Order(DJEMBES, int(djembe_mid), adjusted_qty))
        else:
            orders.append(Order(PICNIC_BASKET1, int(basket1_mid), adjusted_qty))
            orders.append(Order(CROISSANTS, int(croissant_mid), -6 * adjusted_qty))
            orders.append(Order(JAMS, int(jam_mid), -3 * adjusted_qty))
            orders.append(Order(DJEMBES, int(djembe_mid), -adjusted_qty))

    # Basket 2 Arbitrage
    diff2 = abs(basket2_theoretical - basket2_mid)
    if diff2 >= MIN_ARB_DIFF:
        qty2 = min(base_qty, POSITION_LIMITS[PICNIC_BASKET2] - state.position.get(PICNIC_BASKET2, 0))
        scale = SCALING_FACTOR * (diff2 / basket2_mid)
        adjusted_qty2 = max(1, int(qty2 * scale))
        if basket2_theoretical < basket2_mid:
            orders.append(Order(PICNIC_BASKET2, int(basket2_mid), -adjusted_qty2))
            orders.append(Order(CROISSANTS, int(croissant_mid), 4 * adjusted_qty2))
            orders.append(Order(JAMS, int(jam_mid), 2 * adjusted_qty2))
        else:
            orders.append(Order(PICNIC_BASKET2, int(basket2_mid), adjusted_qty2))
            orders.append(Order(CROISSANTS, int(croissant_mid), -4 * adjusted_qty2))
            orders.append(Order(JAMS, int(jam_mid), -2 * adjusted_qty2))

    return orders


def advanced_market_making_strategy(state: TradingState, symbol: str, base_order_size: int = 10) -> list[Order]:
    orders = []
    fair_price = get_mid_price(state, symbol)
    hist = update_mid_price_history(state, symbol)
    vol = estimate_volatility(hist)
    tick_offset = BASE_TICK + int(VOL_SCALE * vol)

    od = state.order_depths[symbol]
    best_bid = max(od.buy_orders.keys()) if od.buy_orders else fair_price - tick_offset
    best_ask = min(od.sell_orders.keys()) if od.sell_orders else fair_price + tick_offset
    buy_price = min(int(fair_price), best_bid + tick_offset)
    sell_price = max(int(fair_price), best_ask - tick_offset)

    current_pos = state.position.get(symbol, 0)
    risk_adjusted_size = max(1, int(base_order_size * (1 - abs(current_pos) / POSITION_LIMITS[symbol])))

    if current_pos < POSITION_LIMITS[symbol]:
        orders.append(Order(symbol, buy_price, risk_adjusted_size))
    if current_pos > -POSITION_LIMITS[symbol]:
        orders.append(Order(symbol, sell_price, -risk_adjusted_size))

    return orders



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
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {symbol: [] for symbol in [CROISSANTS, JAMS, DJEMBES, PICNIC_BASKET1, PICNIC_BASKET2]}
        conversions = 0
        trader_data = "Adaptive MM + Deviation Scaled Arbitrage w/ some typa Risk Adjustments"

        arb_orders = arbitrage_strategy(state, base_qty=MAX_BASE_QTY)
        for o in arb_orders:
            result.setdefault(o.symbol, []).append(o)
            logger.print("Arb order:", o.symbol, o.price, o.quantity)

        for basket in [PICNIC_BASKET1, PICNIC_BASKET2]:
            mm_orders = advanced_market_making_strategy(state, basket, base_order_size=10)
            for o in mm_orders:
                result.setdefault(o.symbol, []).append(o)
                logger.print("MM order for", basket, ":", o.symbol, o.price, o.quantity)

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
