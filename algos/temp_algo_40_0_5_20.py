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
                # observation.transportFees,
                # observation.exportTariff,
                # observation.importTariff,
                # observation.sunlight,
                # observation.humidity,
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

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS" 
    SYNTHETIC = "SYNTHETIC_INTERNAL"
    SPREAD_PB1 = "SPREAD_PB1"
    SPREAD_PB2 = "SPREAD_PB2"

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
        "MAGNIFICENT_MACARONS": 75
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

    _basket_weights_1 = {
        "CROISSANTS": 6,
        "JAMS": 3,
        "DJEMBES": 1
    }

    _basket_weights_2 = {
        "CROISSANTS": 4,
        "JAMS": 2,
    }

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
    
class Trader:
    status_objects: Dict[str, Status] = {}
    state_croiss = Status("CROISSANTS")
    state_jam = Status("JAMS")
    state_djembes = Status("DJEMBES")
    state_picnic1 = Status("PICNIC_BASKET1")

    def get_swmid(self, order_depth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
            best_bid_vol + best_ask_vol
        )
    
    def get_synthetic_basket_order_depth(self, order_depths: Dict[str, OrderDepth]) -> OrderDepth:
        synthetic_order_price = OrderDepth()

        croissants_best_bid = (
            max(order_depths["CROISSANTS"].buy_orders.keys())
            if order_depths["CROISSANTS"].buy_orders else 0)
    
        croissants_best_ask = (
            min(order_depths["CROISSANTS"].sell_orders.keys())
            if order_depths["CROISSANTS"].sell_orders else 0)
        
        jams_best_bid = (
            max(order_depths["JAMS"].buy_orders.keys())
            if order_depths["JAMS"].buy_orders else 0)
    
        jams_best_ask = (
            min(order_depths["JAMS"].sell_orders.keys())
            if order_depths["JAMS"].sell_orders else 0)
        
        djembes_best_bid = (
            max(order_depths["DJEMBES"].buy_orders.keys())
            if order_depths["DJEMBES"].buy_orders else 0)
        
        djembes_best_ask = (
            min(order_depths["DJEMBES"].sell_orders.keys())
            if order_depths["DJEMBES"].sell_orders else 0)
        
        implied_bid = (
            croissants_best_bid * Status._basket_weights_1["CROISSANTS"] +
             jams_best_bid * Status._basket_weights_1["JAMS"] +
             djembes_best_bid * Status._basket_weights_1["DJEMBES"] 
            )

        implied_ask = (
            croissants_best_ask * Status._basket_weights_1["CROISSANTS"] +
            jams_best_ask * Status._basket_weights_1["JAMS"] +
            djembes_best_ask * Status._basket_weights_1["DJEMBES"]
        )

        if implied_bid > 0:
            croissant_bid_volume = (order_depths["CROISSANTS"].buy_orders[croissants_best_bid] // Status._basket_weights_1["CROISSANTS"])
            jams_bid_volume = (order_depths["JAMS"].buy_orders[jams_best_bid] // Status._basket_weights_1["JAMS"])
            djembes_bid_volume = (order_depths["DJEMBES"].buy_orders[djembes_best_bid] // Status._basket_weights_1["DJEMBES"])
            implied_bid_volume = min(croissant_bid_volume, jams_bid_volume, djembes_bid_volume)
            synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume
        if implied_ask > 0:
            croissant_ask_volume = (order_depths["CROISSANTS"].sell_orders[croissants_best_ask] // Status._basket_weights_1["CROISSANTS"])
            jams_ask_volume = (order_depths["JAMS"].sell_orders[jams_best_ask] // Status._basket_weights_1["JAMS"])
            djembes_ask_volume = (order_depths["DJEMBES"].sell_orders[djembes_best_ask] // Status._basket_weights_1["DJEMBES"])
            implied_ask_volume = min(croissant_ask_volume, jams_ask_volume, djembes_ask_volume)
            synthetic_order_price.sell_orders[implied_ask] = implied_ask_volume
        return synthetic_order_price
    
    def convert_synthetic_basket_orders(self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]) -> List[Order]:
        component_orders = {
            "CROISSANTS": [],
            "JAMS": [],
            "DJEMBES": []
        }

        synthetic_basket_order_depth = self.get_synthetic_basket_order_depth(order_depths)

        best_bid = (
            max(synthetic_basket_order_depth.buy_orders.keys())
            if synthetic_basket_order_depth.buy_orders else 0
        )

        best_ask = (
            min(synthetic_basket_order_depth.sell_orders.keys())
            if synthetic_basket_order_depth.sell_orders else 0
        )

        for order in synthetic_orders:
            price = order.price
            quantity = order.quantity

            if quantity > 0 and price >= best_ask:
                croissant_price = min(order_depths["CROISSANTS"].sell_orders.keys())
                jams_price = min(order_depths["JAMS"].sell_orders.keys())
                djembes_price = min(order_depths["DJEMBES"].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                croissant_price = max(order_depths["CROISSANTS"].buy_orders.keys())
                jams_price = max(order_depths["JAMS"].buy_orders.keys())
                djembes_price = max(order_depths["DJEMBES"].buy_orders.keys())
            else:
                continue

            croissant_order = Order(
                "CROISSANTS",
                croissant_price,
                quantity * Status._basket_weights_1["CROISSANTS"]
            )
            jams_order = Order(
                "JAMS",
                jams_price,
                quantity * Status._basket_weights_1["JAMS"]
            )
            djembes_order = Order(
                "DJEMBES",
                djembes_price,
                quantity * Status._basket_weights_1["DJEMBES"]
            )

            component_orders["CROISSANTS"].append(croissant_order)
            component_orders["JAMS"].append(jams_order)
            component_orders["DJEMBES"].append(djembes_order)
        return component_orders
    
    def execute_spread_orders(self, target_position: int, basket_position: int, order_depths: Dict[str, OrderDepth]):
        if target_position == basket_position:
            return []
        
        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths["PICNIC_BASKET1"]
        synthetic_oder_depth = self.get_synthetic_basket_order_depth(order_depths)

        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])
            
            synthetic_bid_price = max(synthetic_oder_depth.buy_orders.keys())
            synthetic_bid_volume = abs(synthetic_oder_depth.buy_orders[synthetic_bid_price])

            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            executed_volume = min(orderbook_volume, target_quantity)

            basket_orders = [Order("PICNIC_BASKET1", basket_ask_price, -executed_volume)]
            synthetic_orders = [Order("SYNTHETIC", synthetic_bid_price, executed_volume)]

            aggregate_orders = self.convert_synthetic_basket_orders(synthetic_orders, order_depths)

            aggregate_orders["PICNIC_BASKET1"] = basket_orders
            return aggregate_orders
        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])
            
            synthetic_ask_price = min(synthetic_oder_depth.sell_orders.keys())
            synthetic_ask_volume = abs(synthetic_oder_depth.sell_orders[synthetic_ask_price])

            order_book_volume = min(basket_bid_volume, synthetic_ask_volume)
            executed_volume = min(order_book_volume, target_quantity)
            basket_orders = [Order("PICNIC_BASKET1", basket_bid_price, -executed_volume)]
            synthetic_orders = [Order("SYNTHETIC", synthetic_ask_price, executed_volume)]

            aggregate_orders = self.convert_synthetic_basket_orders(synthetic_orders, order_depths)

            aggregate_orders["PICNIC_BASKET1"] = basket_orders
            return aggregate_orders
        
    def spread_orders(self, order_depths: Dict[str, OrderDepth], product: Product, basket_position, spread_data: Dict[str, Any]): # Explicitly type spread_data
        # Ensure required keys exist in order_depths to avoid KeyError
        required_keys = ["PICNIC_BASKET1", "CROISSANTS", "JAMS", "DJEMBES"]
        if not all(key in order_depths for key in required_keys):
             logger.print("Missing required order depths for spread calculation.")
             return [] # Return empty dictionary if any key is missing

        # Check if order books are empty (handle potential errors in get_swmid)
        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        if not basket_order_depth.buy_orders or not basket_order_depth.sell_orders:
             logger.print("Basket order book is empty or one-sided.")
             return []

        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths)
        if not synthetic_order_depth.buy_orders or not synthetic_order_depth.sell_orders:
             logger.print("Synthetic order book is empty or one-sided.")
             # Consider if you want to proceed with only basket data or return
             return [] # Example: Return if synthetic cannot be calculated

        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        logger.print(f"Basket SWMID: {basket_swmid}, Synthetic SWMID: {synthetic_swmid}")

        spread = synthetic_swmid - basket_swmid
        logger.print(f"Spread: {spread}")
        spread_data["spread_history"].append(spread)

        # Ensure history has enough data points for calculation
        if len(spread_data["spread_history"]) < self.Params[Product.SPREAD_PB1]["spread_std_window"]:
            logger.print(f"Spread history length ({len(spread_data['spread_history'])}) less than window ({self.Params[Product.SPREAD_PB1]['spread_std_window']}).")
            return [] # Return empty dictionary

        # Maintain the rolling window size
        if len(spread_data["spread_history"]) > self.Params[Product.SPREAD_PB1]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        # --- Calculate rolling mean AND standard deviation ---
        recent_spreads = np.array(spread_data["spread_history"])
        spread_mean_rolling = np.mean(recent_spreads) # Calculate rolling mean
        spread_std_rolling = np.std(recent_spreads)   # Calculate rolling std dev
        logger.print(f"Rolling Spread Mean: {spread_mean_rolling}, Rolling Spread Std: {spread_std_rolling}")

        # Avoid division by zero or very small std dev
        if spread_std_rolling < 1e-6: # Add a small tolerance check
             logger.print("Spread standard deviation is near zero, skipping z-score calculation.")
             return [] # Return empty dictionary

        # --- Calculate z-score using ROLLING mean and std dev ---
        zscore = (spread - spread_mean_rolling) / spread_std_rolling
        logger.print(f"Calculated Z-score: {zscore}")

        # Store the calculated z-score (as float) for the next iteration
        spread_data["prev_zscore"] = float(zscore)

        orders_to_execute = [] # Initialize as an empty list

        # --- Trading Logic based on z-score ---
        target_pos = self.Params[Product.SPREAD_PB1]["target_position"] # Get target size
        threshold = self.Params[Product.SPREAD_PB1]["zscore_threshold"] # Get z-score threshold

        if zscore >= threshold:
             # Spread is high (synthetic expensive relative to basket) -> Sell synthetic, Buy basket
             desired_position = -target_pos # Target position for the basket
             if basket_position != desired_position:
                  logger.print(f"Z-score ({zscore:.2f}) >= threshold ({threshold:.2f}). Target: {desired_position}, Current: {basket_position}. Placing orders.")
                  orders_to_execute = self.execute_spread_orders(desired_position, basket_position, order_depths)
             else:
                  logger.print(f"Z-score ({zscore:.2f}) >= threshold ({threshold:.2f}), but already at target position {basket_position}.")


        elif zscore <= -threshold:
             # Spread is low (synthetic cheap relative to basket) -> Buy synthetic, Sell basket
             desired_position = target_pos # Target position for the basket
             if basket_position != desired_position:
                  logger.print(f"Z-score ({zscore:.2f}) <= threshold ({-threshold:.2f}). Target: {desired_position}, Current: {basket_position}. Placing orders.")
                  orders_to_execute = self.execute_spread_orders(desired_position, basket_position, order_depths)
             else:
                  logger.print(f"Z-score ({zscore:.2f}) <= threshold ({-threshold:.2f}), but already at target position {basket_position}.")

        else:
             # Z-score is within the threshold, potentially close positions or do nothing
             # Example: Close positions if near zero z-score
             if abs(zscore) < 0.1: # Example closing threshold
                  desired_position = 0
                  if basket_position != desired_position:
                       logger.print(f"Z-score ({zscore:.2f}) near zero. Closing position. Target: {desired_position}, Current: {basket_position}.")
                       orders_to_execute = self.execute_spread_orders(desired_position, basket_position, order_depths)
                  else:
                       logger.print(f"Z-score ({zscore:.2f}) within threshold ({threshold:.2f}) and position is zero.")
             else:
                  logger.print(f"Z-score ({zscore:.2f}) within threshold ({threshold:.2f}). No action.")


        return orders_to_execute # Return the list/dict of orders (or empty)

    @staticmethod
    def convert(state: Status):
        if state.position < 0:
            return -state.position
        elif state.position > 0:
            return -state.position
        else:
            return 0
    
    Params = {
        Product.SPREAD_PB1: {
            "spread_mean": 32856.62480438185,
            "spread_std" : 11135.827561425862,
            "spread_std_window" : 40,
            "zscore_threshold" : 0.5,
            "target_position": 20
        },
        Product.SPREAD_PB2: {
            "spread_mean" : 12970.22061803445,
            "spread_std" : 5743.741908271134,
            "spread_std_window" : 10,
            "zscore_threshold" : 0,
            "target_position": 0
        },

    }

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        Status.cls_update(state) # Update global status with the new state

        traderObject = {}
        if state.traderData != None and state.traderData != "":
            try:
                # Attempt to decode the traderData
                traderObject = jsonpickle.decode(state.traderData)
            except Exception as e:
                # Log an error if decoding fails and start with an empty object
                logger.print(f"Error decoding traderData: {e}. Initializing with empty object.")
                traderObject = {}

        result = {}
        conversions = 0

        # --- SPREAD_PB1 Logic ---
        # Ensure SPREAD_PB1 data structure exists (initialize if first run or if missing)
        if Product.SPREAD_PB1 not in traderObject:
            logger.print("Initializing traderObject for SPREAD_PB1")
            traderObject[Product.SPREAD_PB1] = {
                "spread_history": [],
                "prev_zscore": 0.0, # Initialize as float
            }

        # This part now runs EVERY time for SPREAD_PB1
        basket_position_pb1 = state.position.get(Product.PICNIC_BASKET1, 0) # Use .get for safety

        # Call spread_orders to calculate spread, update history, and potentially get orders
        # Pass the specific part of the state for SPREAD_PB1
        spread_orders_dict_pb1 = self.spread_orders(
            state.order_depths,
            Product.PICNIC_BASKET1, # Assuming this indicates which basket to check position for
            basket_position_pb1,
            traderObject[Product.SPREAD_PB1], # Pass the relevant dict
        )
        # logger.print("Spread PB1 orders:", spread_orders_dict_pb1) # Optional: Log returned orders

        # Add returned orders for PB1 to the result
        if spread_orders_dict_pb1: # Check if it's not empty
            for product, orders in spread_orders_dict_pb1.items():
                 if product in result:
                     result[product].extend(orders)
                 else:
                     result[product] = list(orders) # Ensure it's a list

        # --- Placeholder for SPREAD_PB2 Logic (if you implement it) ---
        # Example:
        # if Product.SPREAD_PB2 not in traderObject:
        #     logger.print("Initializing traderObject for SPREAD_PB2")
        #     traderObject[Product.SPREAD_PB2] = { ... } # Initialize similarly
        #
        # basket_position_pb2 = state.position.get(Product.PICNIC_BASKET2, 0)
        # spread_orders_dict_pb2 = self.spread_orders(state.order_depths, Product.PICNIC_BASKET2, basket_position_pb2, traderObject[Product.SPREAD_PB2])
        # logger.print("Spread PB2 orders:", spread_orders_dict_pb2)
        # if spread_orders_dict_pb2:
        #     for product, orders in spread_orders_dict_pb2.items():
        #         # Add or extend orders in the result dictionary
        #         ...

        # --- Placeholder for other product strategies ---
        # Add logic for other products like VOLCANIC_ROCK, MACARONS etc. here
        # e.g., result["VOLCANIC_ROCK"] = self.volcanic_rock_strategy(state, traderObject)

        # Encode the updated traderObject back into traderData string
        traderData = jsonpickle.encode(traderObject)

        # Flush logs and return results
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData