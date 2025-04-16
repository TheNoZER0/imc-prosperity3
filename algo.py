import json
from typing import Any, List, Dict
import jsonpickle
import numpy as np
import pandas as pd
import math
from datamodel import *
import pandas as pd
from statistics import NormalDist

INF = 1e9
EMA_PERIOD = 40
MAX_BASE_QTY = 10

def compute_time_to_expiry(round_number: int, current_timestamp: int) -> float:
    total_round_units = 7_000_000
    day_units = 1_000_000
    remaining_units = total_round_units - (((round_number - 1) * day_units) + current_timestamp)
    return (remaining_units / day_units) / 365.25

def norm_cdf(x: float) -> float:
    a1 = 0.31938153
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429
    p  = 0.2316419
    c  = 0.3989422804014327  # 1/sqrt(2*pi)
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1 / (1 + p * x)
    poly = a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
    result = 1 - c * math.exp(-x**2 / 2) * poly
    return 0.5 * (1 + sign * result)

def bs_coupon_price(spot: float, strike: float, time_to_expiry: float,
                    risk_free_rate: float, volatility: float) -> float:
    """
    Calculate the Black–Scholes price for a voucher (European call option)
    using statistics.NormalDist.
    """
    # Handle edge cases: zero time/volatility or non-positive inputs
    if time_to_expiry <= 1e-9 or volatility <= 1e-9 or spot <= 0 or strike <= 0:
        # Return intrinsic value at expiry, discounted
        return max(spot - strike * math.exp(-risk_free_rate * time_to_expiry), 0.0)

    sqrt_T = math.sqrt(time_to_expiry)
    d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * sqrt_T)
    d2 = d1 - volatility * sqrt_T

    try:
        # Use NormalDist().cdf()
        price = spot * NormalDist().cdf(d1) - strike * math.exp(-risk_free_rate * time_to_expiry) * NormalDist().cdf(d2)
        return price
    except Exception as e:
        # Log error if needed, return NaN or intrinsic value on distribution error
        print(f"Error in NormalDist calculation: {e}, d1={d1}, d2={d2}") # Use logger in actual code
        return max(spot - strike * math.exp(-risk_free_rate * time_to_expiry), 0.0)
    
def calculate_delta(spot: float, strike: float, time_to_expiry: float,
                    risk_free_rate: float, volatility: float) -> float:
    """
    Calculate the Black–Scholes delta for a voucher (European call option)
    using statistics.NormalDist.
    """
    # Handle edge cases
    if time_to_expiry <= 1e-9 or volatility <= 1e-9 or spot <= 0 or strike <= 0:
        # Delta is 0 if OTM/ATM at expiry, 1 if ITM at expiry
        return 1.0 if spot > strike else 0.0

    sqrt_T = math.sqrt(time_to_expiry)
    d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * sqrt_T)

    try:
        # Use NormalDist().cdf()
        delta = NormalDist().cdf(d1)
        return delta
    except Exception as e:
        print(f"Error in NormalDist calculation for delta: {e}, d1={d1}") # Use logger
        # Fallback delta based on intrinsic value
        return 1.0 if spot > strike else 0.0


def implied_volatility(market_price: float, spot: float, strike: float, round_number: int,
                       current_timestamp: int, r: float = 0.0, tol: float = 1e-6, max_iter: int = 1000) -> float:
    TTE = compute_time_to_expiry(round_number, current_timestamp)
    
    def objective(vol):
        return bs_coupon_price(spot, strike, TTE, r, vol) - market_price
    
    lower_bound = 1e-6
    upper_bound = 1.0
    
    if objective(lower_bound) * objective(upper_bound) > 0:
        return np.nan
    
    for _ in range(max_iter):
        mid_vol = (lower_bound + upper_bound) / 2
        if abs(objective(mid_vol)) < tol:
            return mid_vol
        if objective(lower_bound) * objective(mid_vol) < 0:
            upper_bound = mid_vol
        else:
            lower_bound = mid_vol
    
    return np.nan

def compute_m_t(spot: float, strike: float, time_to_expiry: float) -> float:
    if spot <= 0 or time_to_expiry <= 0:
        return np.nan
    return math.log(strike / spot) / math.sqrt(time_to_expiry)

def fit_parabolic(m_values: np.ndarray, v_values: np.ndarray) -> np.ndarray:
    coeffs = np.polyfit(m_values, v_values, 2)
    return coeffs

def evaluate_parabolic(m: float, coeffs: np.ndarray) -> float:
    return coeffs[0] * m**2 + coeffs[1] * m + coeffs[2]

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

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
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

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
                observation.sunlight,
                observation.humidity,
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

        return value[:max_length - 3] + "..."

logger = Logger()

class Status:

    _position_limit = {
        "RAINFOREST_RESIN": 50,
        "KELP": 50,
        "SQUID_INK": 50,
        "CROISSANTS": 250,
        "JAMS": 250,
        "DJEMBES": 60,
        "PICNIC_BASKET1": 60,
        "PICNIC_BASKET2": 100,
        "VOLCANIC_ROCK": 400,
        "VOLCANIC_ROCK_VOUCHER_9500": 200,
        "VOLCANIC_ROCK_VOUCHER_9750": 200,
        "VOLCANIC_ROCK_VOUCHER_10000": 200,
        "VOLCANIC_ROCK_VOUCHER_10250": 200,
        "VOLCANIC_ROCK_VOUCHER_10500": 200,
    }

    _state = None

    _realtime_position = {key:0 for key in _position_limit.keys()}

    _hist_order_depths = {
        product:{
            'bidprc1': [],
            'bidamt1': [],
            'bidprc2': [],
            'bidamt2': [],
            'bidprc3': [],
            'bidamt3': [],
            'askprc1': [],
            'askamt1': [],
            'askprc2': [],
            'askamt2': [],
            'askprc3': [],
            'askamt3': [],
        } for product in _position_limit.keys()
    }

    _hist_observation = {
        'sunlight': [],
        'humidity': [],
        'transportFees': [],
        'exportTariff': [],
        'importTariff': [],
        'bidPrice': [],
        'askPrice': [],
    }

    _num_data = 0

    def __init__(self, product: str) -> None:
        """Initialize status object.

        Args:
            product (str): product

        """
        self.product = product

    @classmethod
    def cls_update(cls, state: TradingState) -> None:
        """Update trading state.

        Args:
            state (TradingState): trading state

        """
        # Update TradingState
        cls._state = state
        # Update realtime position
        for product, posit in state.position.items():
            cls._realtime_position[product] = posit
        # Update historical order_depths
        for product, orderdepth in state.order_depths.items():
            cnt = 1
            for prc, amt in sorted(orderdepth.sell_orders.items(), reverse=False):
                cls._hist_order_depths[product][f'askamt{cnt}'].append(amt)
                cls._hist_order_depths[product][f'askprc{cnt}'].append(prc)
                cnt += 1
                if cnt == 4:
                    break
            while cnt < 4:
                cls._hist_order_depths[product][f'askprc{cnt}'].append(np.nan)
                cls._hist_order_depths[product][f'askamt{cnt}'].append(np.nan)
                cnt += 1
            cnt = 1
            for prc, amt in sorted(orderdepth.buy_orders.items(), reverse=True):
                cls._hist_order_depths[product][f'bidprc{cnt}'].append(prc)
                cls._hist_order_depths[product][f'bidamt{cnt}'].append(amt)
                cnt += 1
                if cnt == 4:
                    break
            while cnt < 4:
                cls._hist_order_depths[product][f'bidprc{cnt}'].append(np.nan)
                cls._hist_order_depths[product][f'bidamt{cnt}'].append(np.nan)
                cnt += 1
        cls._num_data += 1
        
        # cls._hist_observation['sunlight'].append(state.observations.conversionObservations['ORCHIDS'].sunlight)
        # cls._hist_observation['humidity'].append(state.observations.conversionObservations['ORCHIDS'].humidity)
        # cls._hist_observation['transportFees'].append(state.observations.conversionObservations['ORCHIDS'].transportFees)
        # cls._hist_observation['exportTariff'].append(state.observations.conversionObservations['ORCHIDS'].exportTariff)
        # cls._hist_observation['importTariff'].append(state.observations.conversionObservations['ORCHIDS'].importTariff)
        # cls._hist_observation['bidPrice'].append(state.observations.conversionObservations['ORCHIDS'].bidPrice)
        # cls._hist_observation['askPrice'].append(state.observations.conversionObservations['ORCHIDS'].askPrice)

    def hist_order_depth(self, type: str, depth: int, size) -> np.ndarray:
        """Return historical order depth.

        Args:
            type (str): 'bidprc' or 'bidamt' or 'askprc' or 'askamt'
            depth (int): depth, 1 or 2 or 3
            size (int): size of data

        Returns:
            np.ndarray: historical order depth for given type and depth

        """
        return np.array(self._hist_order_depths[self.product][f'{type}{depth}'][-size:], dtype=np.float32)
    
    @property
    def timestep(self) -> int:
        return self._state.timestamp / 100

    @property
    def position_limit(self) -> int:
        """Return position limit of product.

        Returns:
            int: position limit of product

        """
        return self._position_limit[self.product]

    @property
    def position(self) -> int:
        """Return current position of product.

        Returns:
            int: current position of product

        """
        if self.product in self._state.position:
            return int(self._state.position[self.product])
        else:
            return 0
    
    @property
    def rt_position(self) -> int:
        """Return realtime position.

        Returns:
            int: realtime position

        """
        return self._realtime_position[self.product]

    def _cls_rt_position_update(cls, product, new_position):
        if abs(new_position) <= cls._position_limit[product]:
            cls._realtime_position[product] = new_position
        else:
            raise ValueError("New position exceeds position limit")

    def rt_position_update(self, new_position: int) -> None:
        """Update realtime position.

        Args:
            new_position (int): new position

        """
        self._cls_rt_position_update(self.product, new_position)
    
    @property
    def bids(self) -> list[tuple[int, int]]:
        """Return bid orders.

        Returns:
            dict[int, int].items(): bid orders (prc, amt)

        """
        return list(self._state.order_depths[self.product].buy_orders.items())
    
    @property
    def asks(self) -> list[tuple[int, int]]:
        """Return ask orders.

        Returns:
            dict[int, int].items(): ask orders (prc, amt)

        """
        return list(self._state.order_depths[self.product].sell_orders.items())
    
    @classmethod
    def _cls_update_bids(cls, product, prc, new_amt):
        if new_amt > 0:
            cls._state.order_depths[product].buy_orders[prc] = new_amt
        elif new_amt == 0:
            cls._state.order_depths[product].buy_orders[prc] = 0
        # else:
        #     raise ValueError("Negative amount in bid orders")

    @classmethod
    def _cls_update_asks(cls, product, prc, new_amt):
        if new_amt < 0:
            cls._state.order_depths[product].sell_orders[prc] = new_amt
        elif new_amt == 0:
            cls._state.order_depths[product].sell_orders[prc] = 0
        # else:
        #     raise ValueError("Positive amount in ask orders")
        
    def update_bids(self, prc: int, new_amt: int) -> None:
        """Update bid orders.

        Args:
            prc (int): price
            new_amt (int): new amount

        """
        self._cls_update_bids(self.product, prc, new_amt)
    
    def update_asks(self, prc: int, new_amt: int) -> None:
        """Update ask orders.

        Args:
            prc (int): price
            new_amt (int): new amount

        """
        self._cls_update_asks(self.product, prc, new_amt)

    @property
    def possible_buy_amt(self) -> int:
        """Return possible buy amount.

        Returns:
            int: possible buy amount
        
        """
        possible_buy_amount1 = self._position_limit[self.product] - self.rt_position
        possible_buy_amount2 = self._position_limit[self.product] - self.position
        return min(possible_buy_amount1, possible_buy_amount2)
        
    @property
    def possible_sell_amt(self) -> int:
        """Return possible sell amount.

        Returns:
            int: possible sell amount
        
        """
        possible_sell_amount1 = self._position_limit[self.product] + self.rt_position
        possible_sell_amount2 = self._position_limit[self.product] + self.position
        return min(possible_sell_amount1, possible_sell_amount2)

    def hist_mid_prc(self, size:int) -> np.ndarray:
        """Return historical mid price.

        Args:
            size (int): size of data

        Returns:
            np.ndarray: historical mid price
        
        """
        return (self.hist_order_depth('bidprc', 1, size) + self.hist_order_depth('askprc', 1, size)) / 2
    
    def hist_maxamt_askprc(self, size:int) -> np.ndarray:
        """Return price of ask order with maximum amount in historical order depth.

        Args:
            size (int): size of data

        Returns:
            int: price of ask order with maximum amount in historical order depth
        
        """
        res_array = np.empty(size)
        prc_array = np.array([self.hist_order_depth('askprc', 1, size), self.hist_order_depth('askprc', 2, size), self.hist_order_depth('askprc', 3, size)]).T
        amt_array = np.array([self.hist_order_depth('askamt', 1, size), self.hist_order_depth('askamt', 2, size), self.hist_order_depth('askamt', 3, size)]).T

        for i, amt_arr in enumerate(amt_array):
            res_array[i] = prc_array[i,np.nanargmax(amt_arr)]

        return res_array

    def hist_maxamt_bidprc(self, size:int) -> np.ndarray:
        """Return price of ask order with maximum amount in historical order depth.

        Args:
            size (int): size of data

        Returns:
            int: price of ask order with maximum amount in historical order depth
        
        """
        res_array = np.empty(size)
        prc_array = np.array([self.hist_order_depth('bidprc', 1, size), self.hist_order_depth('bidprc', 2, size), self.hist_order_depth('bidprc', 3, size)]).T
        amt_array = np.array([self.hist_order_depth('bidamt', 1, size), self.hist_order_depth('bidamt', 2, size), self.hist_order_depth('bidamt', 3, size)]).T

        for i, amt_arr in enumerate(amt_array):
            res_array[i] = prc_array[i,np.nanargmax(amt_arr)]

        return res_array
    
    def hist_vwap_all(self, size:int) -> np.ndarray:
        res_array = np.zeros(size)
        volsum_array = np.zeros(size)
        for i in range(1,4):
            tmp_bid_vol = self.hist_order_depth(f'bidamt', i, size)
            tmp_ask_vol = self.hist_order_depth(f'askamt', i, size)
            tmp_bid_prc = self.hist_order_depth(f'bidprc', i, size)
            tmp_ask_prc = self.hist_order_depth(f'askprc', i, size)
            if i == 0:
                res_array = res_array + (tmp_bid_prc*tmp_bid_vol) + (-tmp_ask_prc*tmp_ask_vol)
                volsum_array = volsum_array + tmp_bid_vol - tmp_ask_vol
            else:
                bid_nan_idx = np.isnan(tmp_bid_prc)
                ask_nan_idx = np.isnan(tmp_ask_prc)
                res_array = res_array + np.where(bid_nan_idx, 0, tmp_bid_prc*tmp_bid_vol) + np.where(ask_nan_idx, 0, -tmp_ask_prc*tmp_ask_vol)
                volsum_array = volsum_array + np.where(bid_nan_idx, 0, tmp_bid_vol) - np.where(ask_nan_idx, 0, tmp_ask_vol)
                
        return res_array / volsum_array
    @property
    def best_bid(self) -> int:
        """Return best bid price and amount.

        Returns:
            tuple[int, int]: (price, amount)
        
        """
        buy_orders = self._state.order_depths[self.product].buy_orders
        if len(buy_orders) > 0:
            return max(buy_orders.keys())
        else:
            return self.best_ask - 1

    @property
    def best_ask(self) -> int:
        sell_orders = self._state.order_depths[self.product].sell_orders
        if len(sell_orders) > 0:
            return min(sell_orders.keys())
        else:
            return self.best_bid + 1

    @property
    def mid(self) -> float:
        return (self.best_bid + self.best_ask) / 2
    
    @property
    def bid_ask_spread(self) -> int:
        return self.best_ask - self.best_bid

    @property
    def best_bid_amount(self) -> int:
        """Return best bid price and amount.

        Returns:
            tuple[int, int]: (price, amount)
        
        """
        best_prc = max(self._state.order_depths[self.product].buy_orders.keys())
        best_amt = self._state.order_depths[self.product].buy_orders[best_prc]
        return best_amt
        
    @property
    def best_ask_amount(self) -> int:
        """Return best ask price and amount.

        Returns:
            tuple[int, int]: (price, amount)
        
        """
        best_prc = min(self._state.order_depths[self.product].sell_orders.keys())
        best_amt = self._state.order_depths[self.product].sell_orders[best_prc]
        return -best_amt
    
    @property
    def worst_bid(self) -> int:
        buy_orders = self._state.order_depths[self.product].buy_orders
        if len(buy_orders) > 0:
            return min(buy_orders.keys())
        else:
            return self.best_ask - 1

    @property
    def worst_ask(self) -> int:
        sell_orders = self._state.order_depths[self.product].sell_orders
        if len(sell_orders) > 0:
            return max(sell_orders.keys())
        else:
            return self.best_bid + 1

    @property
    def vwap(self) -> float:
        vwap = 0
        total_amt = 0

        for prc, amt in self._state.order_depths[self.product].buy_orders.items():
            vwap += (prc * amt)
            total_amt += amt

        for prc, amt in self._state.order_depths[self.product].sell_orders.items():
            vwap += (prc * abs(amt))
            total_amt += abs(amt)

        vwap /= total_amt
        return vwap

    @property
    def vwap_bidprc(self) -> float:
        """Return volume weighted average price of bid orders.

        Returns:
            float: volume weighted average price of bid orders

        """
        vwap = 0
        for prc, amt in self._state.order_depths[self.product].buy_orders.items():
            vwap += (prc * amt)
        vwap /= sum(self._state.order_depths[self.product].buy_orders.values())
        return vwap
    
    @property
    def vwap_askprc(self) -> float:
        """Return volume weighted average price of ask orders.

        Returns:
            float: volume weighted average price of ask orders

        """
        vwap = 0
        for prc, amt in self._state.order_depths[self.product].sell_orders.items():
            vwap += (prc * -amt)
        vwap /= -sum(self._state.order_depths[self.product].sell_orders.values())
        return vwap

    @property
    def maxamt_bidprc(self) -> int:
        """Return price of bid order with maximum amount.
        
        Returns:
            int: price of bid order with maximum amount

        """
        prc_max_mat, max_amt = 0,0
        for prc, amt in self._state.order_depths[self.product].buy_orders.items():
            if amt > max_amt:
                max_amt = amt
                prc_max_mat = prc

        return prc_max_mat
    
    @property
    def maxamt_askprc(self) -> int:
        """Return price of ask order with maximum amount.

        Returns:
            int: price of ask order with maximum amount
        
        """
        prc_max_mat, max_amt = 0,0
        for prc, amt in self._state.order_depths[self.product].sell_orders.items():
            if amt < max_amt:
                max_amt = amt
                prc_max_mat = prc

        return prc_max_mat
    
    @property
    def maxamt_midprc(self) -> float:
        return (self.maxamt_bidprc + self.maxamt_askprc) / 2

    def bidamt(self, price) -> int:
        order_depth = self._state.order_depths[self.product].buy_orders
        if price in order_depth.keys():
            return order_depth[price]
        else:
            return 0
        
    def askamt(self, price) -> int:
        order_depth = self._state.order_depths[self.product].sell_orders
        if price in order_depth.keys():
            return order_depth[price]
        else:
            return 0

class StrategyParameters:
    # Store pre-calculated values here
    PB1_INTERCEPT = 39630.2076
    PB1_HEDGE_C = 0.8869
    PB1_HEDGE_J = 1.1386
    PB1_HEDGE_D = 0.5892
    PB1_SPREAD_STD_DEV = 150.0
    PB1_ENTRY_THRESHOLD = 0.75
    PB1_EXIT_THRESHOLD = 0.1
    TRADE_SIZE_PB1 = 60

    LIMITS = Status._position_limit

class Strategy:
    @staticmethod
    def arb(state: Status, fair_price):
        orders = []

        for ask_price, ask_amount in state.asks:
            if ask_price < fair_price:
                buy_amount = min(-ask_amount, state.possible_buy_amt)
                if buy_amount > 0:
                    orders.append(Order(state.product, int(ask_price), int(buy_amount)))
                    state.rt_position_update(state.rt_position + buy_amount)
                    state.update_asks(ask_price, -(-ask_amount - buy_amount))

            elif ask_price == fair_price:
                if state.rt_position < 0:
                    buy_amount = min(-ask_amount, -state.rt_position)
                    orders.append(Order(state.product, int(ask_price), int(buy_amount)))
                    state.rt_position_update(state.rt_position + buy_amount)
                    state.update_asks(ask_price, -(-ask_amount - buy_amount))

        for bid_price, bid_amount in state.bids:
            if bid_price > fair_price:
                sell_amount = min(bid_amount, state.possible_sell_amt)
                if sell_amount > 0:
                    orders.append(Order(state.product, int(bid_price), -int(sell_amount)))
                    state.rt_position_update(state.rt_position - sell_amount)
                    state.update_bids(bid_price, bid_amount - sell_amount)

            elif bid_price == fair_price:
                if state.rt_position > 0:
                    sell_amount = min(bid_amount, state.rt_position)
                    orders.append(Order(state.product, int(bid_price), -int(sell_amount)))
                    state.rt_position_update(state.rt_position - sell_amount)
                    state.update_bids(bid_price, bid_amount - sell_amount)

        return orders
    
    @staticmethod
    def mm_glft(
        state: Status,
        fair_price,
        mu,
        sigma,
        gamma=1e-9,
        order_amount=50,
    ):
        q = state.rt_position / order_amount

        kappa_b = 1 / max((fair_price - state.best_bid) - 1, 1)
        kappa_a = 1 / max((state.best_ask - fair_price) - 1, 1)

        A_b = 0.25
        A_a = 0.25

        delta_b = 1 / gamma * math.log(1 + gamma / kappa_b) + (-mu / (gamma * sigma**2) + (2 * q + 1) / 2) * math.sqrt((sigma**2 * gamma) / (2 * kappa_b * A_b) * (1 + gamma / kappa_b)**(1 + kappa_b / gamma))
        delta_a = 1 / gamma * math.log(1 + gamma / kappa_a) + (mu / (gamma * sigma**2) - (2 * q - 1) / 2) * math.sqrt((sigma**2 * gamma) / (2 * kappa_a * A_a) * (1 + gamma / kappa_a)**(1 + kappa_a / gamma))

        p_b = round(fair_price - delta_b)
        p_a = round(fair_price + delta_a)

        p_b = min(p_b, fair_price) # Set the buy price to be no higher than the fair price to avoid losses
        p_b = min(p_b, state.best_bid + 1) # Place the buy order as close as possible to the best bid price
        p_b = max(p_b, state.maxamt_bidprc + 1) # No market order arrival beyond this price

        p_a = max(p_a, fair_price)
        p_a = max(p_a, state.best_ask - 1)
        p_a = min(p_a, state.maxamt_askprc - 1)

        buy_amount = min(order_amount, state.possible_buy_amt)
        sell_amount = min(order_amount, state.possible_sell_amt)

        orders = []
        if buy_amount > 0:
            orders.append(Order(state.product, int(p_b), int(buy_amount)))
        if sell_amount > 0:
            orders.append(Order(state.product, int(p_a), -int(sell_amount)))
        return orders
    
    @staticmethod
    def mm_ou(
        state: Status,
        fair_price,
        gamma=1e-9,
        order_amount=0,
    ):

        q = state.rt_position / order_amount
        Q = state.position_limit / order_amount

        kappa_b = 1 / max((fair_price - state.best_bid) - 1, 1)
        kappa_a = 1 / max((state.best_ask - fair_price) - 1, 1)
            
        vfucn = lambda q,Q: -INF if (q==Q+1 or q==-(Q+1)) else math.log(math.sin(((q+Q+1)*math.pi)/(2*Q+2)))

        delta_b = 1 / gamma * math.log(1 + gamma / kappa_b) - 1 / kappa_b * (vfucn(q + 1, Q) - vfucn(q, Q))
        delta_a = 1 / gamma * math.log(1 + gamma / kappa_a) + 1 / kappa_a * (vfucn(q, Q) - vfucn(q - 1, Q))

        p_b = round(fair_price - delta_b)
        p_a = round(fair_price + delta_a)

        p_b = min(p_b, fair_price) # Set the buy price to be no higher than the fair price to avoid losses
        p_b = min(p_b, state.best_bid + 1) # Place the buy order as close as possible to the best bid price
        p_b = max(p_b, state.maxamt_bidprc + 1) # No market order arrival beyond this price

        p_a = max(p_a, fair_price)
        p_a = max(p_a, state.best_ask - 1)
        p_a = min(p_a, state.maxamt_askprc - 1)

        buy_amount = min(order_amount, state.possible_buy_amt)
        sell_amount = min(order_amount, state.possible_sell_amt)

        orders = []
        if buy_amount > 0:
            orders.append(Order(state.product, int(p_b), int(buy_amount)))
        if sell_amount > 0:
            orders.append(Order(state.product, int(p_a), -int(sell_amount)))
        return orders
    
    @staticmethod
    def check_limits(current_positions: Dict[str, int], trades: Dict[str, int], limits: Dict[str, int]):
        """ Checks if a potential multi-leg trade respects all position limits """
        for product, trade_qty in trades.items():
            current_pos = current_positions.get(product, 0)
            limit = limits.get(product, 0)
            if limit == 0: continue
            if abs(current_pos + trade_qty) > limit:
                logger.print(f"Limit Check FAIL: Prod={product}, Curr={current_pos}, Trade={trade_qty}, Limit={limit}")
                return False
        return True

    @staticmethod
    def stat_arb(state_pb1: Status, state_c: Status, state_j: Status, state_d: Status) -> Dict[str, List[Order]]:
        orders: Dict[str, List[Order]] = {}
        params = StrategyParameters
        mid_pb1 = state_pb1.mid
        mid_c = state_c.mid
        mid_j = state_j.mid
        mid_d = state_d.mid
        best_bid_pb1 = state_pb1.best_bid
        best_ask_pb1 = state_pb1.best_ask
        best_bid_c = state_c.best_bid; best_ask_c = state_c.best_ask
        best_bid_j = state_j.best_bid; best_ask_j = state_j.best_ask
        best_bid_d = state_d.best_bid; best_ask_d = state_d.best_ask

        predicted_pb1 = (params.PB1_INTERCEPT + params.PB1_HEDGE_C * mid_c + params.PB1_HEDGE_J * mid_j + params.PB1_HEDGE_D * mid_d)
        current_spread = mid_pb1 - predicted_pb1
        current_zscore = current_spread / params.PB1_SPREAD_STD_DEV

        pos_pb1 = state_pb1.rt_position
        pos_c = state_c.rt_position
        pos_j = state_j.rt_position
        pos_d = state_d.rt_position
        current_positions = { state_pb1.product: pos_pb1, state_c.product: pos_c, state_j.product: pos_j, state_d.product: pos_d}
        target_trades: Dict[str, int] = {}

        if current_zscore > params.PB1_ENTRY_THRESHOLD:
            target_trades[state_pb1.product] = -params.TRADE_SIZE_PB1
            target_trades[state_c.product] = round(params.TRADE_SIZE_PB1 * params.PB1_HEDGE_C)
            target_trades[state_j.product] = round(params.TRADE_SIZE_PB1 * params.PB1_HEDGE_J)
            target_trades[state_d.product] = round(params.TRADE_SIZE_PB1 * params.PB1_HEDGE_D)
        elif current_zscore < -params.PB1_ENTRY_THRESHOLD:
            target_trades[state_pb1.product] = params.TRADE_SIZE_PB1
            target_trades[state_c.product] = -round(params.TRADE_SIZE_PB1 * params.PB1_HEDGE_C)
            target_trades[state_j.product] = -round(params.TRADE_SIZE_PB1 * params.PB1_HEDGE_J)
            target_trades[state_d.product] = -round(params.TRADE_SIZE_PB1 * params.PB1_HEDGE_D)
        
        if target_trades:
            if Strategy.check_limits(current_positions, target_trades, params.LIMITS):
                successful_updates = True
                for product, qty in target_trades.items():
                    if qty == 0: continue
                    if product == state_pb1.product: state_obj = state_pb1
                    elif product == state_c.product: state_obj = state_c
                    elif product == state_j.product: state_obj = state_j
                    elif product == state_d.product: state_obj = state_d

                    price = best_ask = state_obj.best_ask if qty > 0 else state_obj.best_bid
                    orders.setdefault(product, []).append(Order(product, price, qty))
        return orders
        
    @staticmethod
    def convert(state: Status):
        if state.position < 0:
            return -state.position
        elif state.position > 0:
            return -state.position
        else:
            return 0
        
# class Trade:
    

    
class Trader:
    status_objects: Dict[str, Status] = {}
    params = StrategyParameters()
    state_croiss = Status("CROISSANTS")
    state_jam = Status("JAMS")
    state_djembes = Status("DJEMBES")
    state_picnic1 = Status("PICNIC_BASKET1")
    state_picnic2 = Status("PICNIC_BASKET2")

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        Status.cls_update(state)

        result = {}
        for product in state.listings:
             if product not in self.status_objects:
                  self.status_objects[product] = Status(product)
        result: Dict[str, List[Order]] = {}
        conversions = 0
        traderData = ""

        pb1_status = self.status_objects.get('PICNIC_BASKET1')
        c_status = self.status_objects.get('CROISSANTS')
        j_status = self.status_objects.get('JAMS')
        d_status = self.status_objects.get('DJEMBES')

        if all([pb1_status, c_status, j_status, d_status]): # Check if all needed Status objects exist
            pb1_arb_orders = Strategy.stat_arb(pb1_status, c_status, j_status, d_status)
            for product, order_list in pb1_arb_orders.items():
                result.setdefault(product, []).extend(order_list)

        conversions = 0 
        traderData = ""
        try:
            final_trader_data = jsonpickle.encode(traderData)
        except Exception as e:
            logger.print(f"Error encoding traderData: {e}")

        logger.flush(state, result, conversions, final_trader_data)
        return result, conversions, final_trader_data