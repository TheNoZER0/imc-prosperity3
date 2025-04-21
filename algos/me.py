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
    
    def compute_weighted_vwap(order_depths, weights):
        """Compute a VWAP across multiple products, weighted by `weights`."""
        num = den = 0
        for prod, w in weights.items():
            od = order_depths[prod]
            vwap = (
                sum(p * q for p, q in od.buy_orders.items())
                + sum(p * abs(q) for p, q in od.sell_orders.items())
            ) / (
                sum(od.buy_orders.values())
                + sum(abs(q) for q in od.sell_orders.values())
            )
            num += w * vwap
            den += w
        return num / den


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

    def compute_realized_pnl(self, trades: List[Trade]) -> float:
        pnl = 0.0
        open_buys: Dict[str, List[Tuple[int, float]]] = {}
        for t in trades:
            sym, qty, price = t.symbol, t.quantity, t.price
            if qty > 0:
                open_buys.setdefault(sym, []).append([qty, price])
            else:
                sell_qty = -qty
                queue = open_buys.get(sym, [])
                while sell_qty > 0 and queue:
                    buy_qty, buy_price = queue[0]
                    match_qty = min(buy_qty, sell_qty)
                    pnl += match_qty * (price - buy_price)
                    buy_qty  -= match_qty
                    sell_qty -= match_qty
                    if buy_qty == 0:
                        queue.pop(0)
                    else:
                        queue[0][0] = buy_qty
        return pnl

    def get_swmid(self, order_depth: OrderDepth) -> float:
        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        bid_vol = abs(order_depth.buy_orders[best_bid])
        ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)
    
    def get_synthetic_basket_order_depth(self, order_depths: Dict[str, OrderDepth]) -> OrderDepth:
        synthetic_order_book = OrderDepth()

        # find the best prices
        cb = order_depths["CROISSANTS"]
        jb = order_depths["JAMS"]
        db = order_depths["DJEMBES"]

        croissant_bid = max(cb.buy_orders.keys()) if cb.buy_orders else 0
        croissant_ask = min(cb.sell_orders.keys()) if cb.sell_orders else 0

        jams_bid = max(jb.buy_orders.keys()) if jb.buy_orders else 0
        jams_ask = min(jb.sell_orders.keys()) if jb.sell_orders else 0

        djembes_bid = max(db.buy_orders.keys()) if db.buy_orders else 0
        djembes_ask = min(db.sell_orders.keys()) if db.sell_orders else 0

        # weighted mid prices
        implied_bid = (
            croissant_bid * Status._basket_weights_1["CROISSANTS"] +
            jams_bid      * Status._basket_weights_1["JAMS"] +
            djembes_bid   * Status._basket_weights_1["DJEMBES"]
        )
        implied_ask = (
            croissant_ask * Status._basket_weights_1["CROISSANTS"] +
            jams_ask      * Status._basket_weights_1["JAMS"] +
            djembes_ask   * Status._basket_weights_1["DJEMBES"]
        )

        # ——— BID SIDE ———
        if implied_bid > 0:
            raw_cb         = abs(cb.buy_orders[croissant_bid])
            w_cb           = Status._basket_weights_1["CROISSANTS"]
            full_baskets_cb = raw_cb // w_cb
            croissant_bid_vol = full_baskets_cb if full_baskets_cb >= 1 else 0


            raw_jb          = abs(jb.buy_orders[jams_bid])
            w_jb            = Status._basket_weights_1["JAMS"]
            full_baskets_jb = raw_jb // w_jb
            jams_bid_vol    = full_baskets_jb if full_baskets_jb >= 1 else 0


            raw_db           = abs(db.buy_orders[djembes_bid])
            w_db             = Status._basket_weights_1["DJEMBES"]
            full_baskets_db  = raw_db // w_db
            djembes_bid_vol  = full_baskets_db if full_baskets_db >= 1 else 0

            implied_bid_vol = min(croissant_bid_vol, jams_bid_vol, djembes_bid_vol)
            synthetic_order_book.buy_orders[implied_bid] = implied_bid_vol


        # ——— ASK SIDE ———
        if implied_ask > 0:
            raw_ca           = abs(cb.sell_orders[croissant_ask])
            w_ca             = Status._basket_weights_1["CROISSANTS"]
            full_baskets_ca  = raw_ca // w_ca
            croissant_ask_vol = full_baskets_ca if full_baskets_ca >= 1 else 0

            raw_ja           = abs(jb.sell_orders[jams_ask])
            w_ja             = Status._basket_weights_1["JAMS"]
            full_baskets_ja  = raw_ja // w_ja
            jams_ask_vol     = full_baskets_ja if full_baskets_ja >= 1 else 0

            raw_da            = abs(db.sell_orders[djembes_ask])
            w_da              = Status._basket_weights_1["DJEMBES"]
            full_baskets_da   = raw_da // w_da
            djembes_ask_vol   = full_baskets_da if full_baskets_da >= 1 else 0

            implied_ask_vol = min(croissant_ask_vol, jams_ask_vol, djembes_ask_vol)
            synthetic_order_book.sell_orders[implied_ask] = implied_ask_vol
        
        return synthetic_order_book


    
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
        
    def spread_orders(self,
                    order_depths: Dict[str, OrderDepth],
                    product: str,
                    basket_position: int,
                    spread_data: Dict[str, Any]) -> List[Order]:

        # … your existing early exits & liquidity checks …

        # build synthetic book
        s_od = self.get_synthetic_basket_order_depth(order_depths)
        b_od = order_depths[product]

        # 3) compute the spread properly
        # basket mid
        basket_mid    = (max(b_od.buy_orders) + min(b_od.sell_orders)) / 2
        # synthetic mid
        synth_mid     = self.get_swmid(s_od)
        spread        = synth_mid - basket_mid

        # 4) append & maintain rolling history
        sh = spread_data.setdefault("spread_history", [])
        sh.append(spread)
        win = self.Params[Product.SPREAD_PB1]["spread_std_window"]
        if len(sh) > win:
            sh.pop(0)
        if len(sh) < win:
            logger.print(f"Need {win} points, have {len(sh)}")
            return []

        # 5) rolling stats & z‑score
        arr   = np.array(sh)
        mu    = arr.mean()
        sigma = arr.std()
        if sigma < 1e-6:
            logger.print("σ≈0, skipping")
            return []

        # compute and name it zscore
        zscore = (spread - mu) / sigma
        logger.print(f"μ={mu:.4f}, σ={sigma:.4f}, z={zscore:.4f}")

        # 6) decide direction (use zscore here, not z)
        thresh = self.Params[Product.SPREAD_PB1]["zscore_threshold"]
        tgt    = self.Params[Product.SPREAD_PB1]["target_position"]
        orders = []

        if zscore >= thresh:
            # synthetic ↑ expensive ⇒ long basket (+tgt), short synthetic
            if basket_position != tgt:
                logger.print(f"z≥{thresh}, going LONG basket → {tgt}")
                orders = self.execute_spread_orders(tgt, basket_position, order_depths)

        elif zscore <= -thresh:
            # synthetic ↓ cheap ⇒ short basket (–tgt), long synthetic
            if basket_position != -tgt:
                logger.print(f"z≤-{thresh}, going SHORT basket → {-tgt}")
                orders = self.execute_spread_orders(-tgt, basket_position, order_depths)

        else:
            # near zero, optionally flatten
            if abs(zscore) < 0.1 and basket_position != 0:
                logger.print("z≈0, closing spread pos")
                orders = self.execute_spread_orders(0, basket_position, order_depths)
            else:
                logger.print(f"|z|<{thresh}, no action")


        # 7) save for next time
        spread_data["prev_zscore"] = float(zscore)
        return orders


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
            "spread_std_window" : 10,
            "zscore_threshold" : 1.2,
            "target_position": 10
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

        # merge any PB1 spread orders into the main result
        if spread_orders_dict_pb1:
            for symbol, orders in spread_orders_dict_pb1.items():
                # make sure we have a list for this symbol
                result.setdefault(symbol, [])
                # append all your generated Order objects
                result[symbol].extend(orders)

        # logger.print("Spread PB1 orders:", spread_orders_dict_pb1) # Optional: Log returned orders


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
        for sym in ["CROISSANTS", "JAMS", "DJEMBES"]:
            od = state.order_depths[sym]
            if od.buy_orders and od.sell_orders:
                bid = max(od.buy_orders)
                ask = min(od.sell_orders)
                size = min(5, Status(sym).possible_buy_amt)
                result.setdefault(sym, []).append(Order(sym, bid,  size))
                result.setdefault(sym, []).append(Order(sym, ask, -size))

        traderData = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, traderData)
        
        return result, conversions, traderData