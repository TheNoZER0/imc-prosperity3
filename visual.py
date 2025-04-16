import json
from typing import Any, List, Dict
import jsonpickle
from scipy.stats import norm
import numpy as np
import pandas as pd
import math
from datamodel import *
import pandas as pd

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
    Calculate the Black–Scholes price for a voucher (European call option).
    """
    if time_to_expiry <= 0 or volatility <= 0:
        return max(spot - strike, 0.0)
    sqrt_T = math.sqrt(time_to_expiry)
    d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * sqrt_T)
    d2 = d1 - volatility * sqrt_T
    price = spot * norm_cdf(d1) - strike * math.exp(-risk_free_rate * time_to_expiry) * norm_cdf(d2)
    return price

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

voucher_m_history: List[float] = []
voucher_v_history: List[float] = []

def compute_voucher_trade(voucher_state, strike: float, round_number: int, current_timestamp: int,
                          r: float = 0.0, vol_multiplier: float = 1.2,
                          base_buffer: float = 5.0, hedge_scale: float = 0.5) -> List[Order]:
    S = voucher_state.mid
    hist = voucher_state.hist_mid_prc(50)
    sigma_hist = np.std(hist)
    if sigma_hist <= 0:
        sigma_hist = 0.01
    volatility = sigma_hist * vol_multiplier
    TTE = compute_time_to_expiry(round_number, current_timestamp)
    
    d1 = (math.log(S / strike) + (r + 0.5 * volatility**2) * TTE) / (volatility * math.sqrt(TTE))
    d2 = d1 - volatility * math.sqrt(TTE)
    fair_value = bs_coupon_price(S, strike, TTE, r, volatility)
    
    current_m = compute_m_t(S, strike, TTE)
    # use the market voucher price as a proxy for vₜ.
    current_v = implied_volatility(S, S, strike, round_number, current_timestamp)
    voucher_m_history.append(current_m)
    voucher_v_history.append(current_v)
    
    # if enough history is available, fit a quadratic curve
    if len(voucher_m_history) >= 20:
        m_array = np.array(voucher_m_history[-20:])
        v_array = np.array(voucher_v_history[-20:])
        coeffs = fit_parabolic(m_array, v_array)
        base_IV = math.log(coeffs[2])  # Base IV is the constant term (at m = 0)
        logger.print(f"Fitted Parabola Coeffs: {coeffs}, Base IV = {base_IV:.2f}")
    else:
        base_IV = None
    
    buffer = base_buffer * volatility
    target_buy = fair_value - buffer
    target_sell = fair_value + buffer

    logger.print(f"[Voucher Trade] {voucher_state.product}: S = {S:.2f}, Fair Value = {fair_value:.2f}, "
                 f"Target Buy = {target_buy:.2f}, Target Sell = {target_sell:.2f}, Delta = {norm_cdf(d1):.2f}")
    
    orders = []
    if S < target_buy:
        qty = voucher_state.possible_buy_amt
        if qty > 0:
            orders.append(Order(voucher_state.product, int(S), qty))
            hedge_qty = int(hedge_scale * norm_cdf(d1) * qty)
            if hedge_qty > 0:
                orders.append(Order("VOLCANIC_ROCK", int(voucher_state.mid), -hedge_qty))
            logger.print(f"Voucher BUY: {voucher_state.product} qty = {qty}, hedge_qty = {hedge_qty}")
    elif S > target_sell:
        qty = voucher_state.possible_sell_amt
        if qty > 0:
            orders.append(Order(voucher_state.product, int(S), -qty))
            hedge_qty = int(hedge_scale * norm_cdf(d1) * qty)
            if hedge_qty > 0:
                orders.append(Order("VOLCANIC_ROCK", int(voucher_state.mid), hedge_qty))
            logger.print(f"Voucher SELL: {voucher_state.product} qty = {qty}, hedge_qty = {hedge_qty}")
    return orders

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
    def index_arb(
        basket: Status,
        jam: Status,
        djembes: Status,
        croissant: Status,
        theta=0,
        threshold=0,
        jam_m= 3,
        croiss_m= 6,
        djembe_m= 1,
    ):
        
        basket_prc = basket.mid
        underlying_prc = jam_m * jam.vwap + croiss_m * croissant.vwap + djembe_m * djembes.vwap
        spread = basket_prc - underlying_prc
        norm_spread = spread - theta

        orders = []
        if norm_spread > threshold:
            orders.append(Order(basket.product, int(basket.worst_bid), -int(basket.possible_sell_amt)))
        elif norm_spread < -threshold:
            orders.append(Order(basket.product, int(basket.worst_ask), int(basket.possible_buy_amt)))

        return orders
    
    @staticmethod
    def pair_trade(croissant: Status, 
                   djembes: Status, 
                   pairs_mu = 267.613375701525, 
                   theta = 1.03482227e+03, 
                   sigma = 4.46392304e-03, 
                   threshold=1 , coint_vec= np.array([0.04234083, -0.07142774])):
        hedge_ratio = abs(coint_vec[0] / coint_vec[1])

        djembes_prc = djembes.vwap
        croissant_prc = croissant.vwap
        spread = croissant_prc + hedge_ratio * djembes_prc
        norm_spread = spread - pairs_mu
        threshold = 1
        croissant_pos = croissant.position
        djembes_pos = djembes.position

        orders = []
        if norm_spread > threshold: 
            if not (croissant_pos < 0 and djembes_pos > 0): 
                sell_qty = int(croissant.possible_sell_amt)
                buy_qty = int(djembes.possible_buy_amt)
                if sell_qty > 0 and buy_qty > 0:
                     orders.append(Order(croissant.product, int(croissant.worst_bid), -sell_qty)) 
                     orders.append(Order(djembes.product, int(djembes.worst_ask), buy_qty))       
    
        elif norm_spread < -threshold: 
            if not (croissant_pos > 0 and djembes_pos < 0):
                 buy_qty = int(croissant.possible_buy_amt)
                 sell_qty = int(djembes.possible_sell_amt)
                 if buy_qty > 0 and sell_qty > 0:
                      orders.append(Order(croissant.product, int(croissant.worst_ask), buy_qty))  
                      orders.append(Order(djembes.product, int(djembes.worst_bid), -sell_qty))     
        else: 
            if croissant_pos > 0 and djembes_pos < 0 and norm_spread >= 0: 
                orders.append(Order(croissant.product, int(croissant.best_bid), -croissant_pos)) # Sell current long position
                orders.append(Order(djembes.product, int(djembes.best_ask), abs(djembes_pos)))   # Buy back current short position
            
            elif croissant_pos < 0 and djembes_pos > 0 and norm_spread <= 0: 
                orders.append(Order(croissant.product, int(croissant.best_ask), abs(croissant_pos))) # Buy back current short position
                orders.append(Order(djembes.product, int(djembes.best_bid), -djembes_pos))     # Sell current long position
        return orders
        
    @staticmethod
    def convert(state: Status):
        if state.position < 0:
            return -state.position
        elif state.position > 0:
            return -state.position
        else:
            return 0
        
    @staticmethod
    def voucher_trade(voucher_state: 'Status', strike: float, round_number: int, current_timestamp: int,
                        r: float = 0.0, vol_multiplier: float = 1.2,
                        base_buffer: float = 5.0, hedge_scale: float = 0.5) -> List[Order]:
        return compute_voucher_trade(voucher_state, strike, round_number, current_timestamp,
                                     r, vol_multiplier, base_buffer, hedge_scale)

CROISSANTS = "CROISSANTS"
EMA_PERIOD = 13       # Window period for the EMA calculation.
class Trade:
    mid_price_history = {CROISSANTS: []}
    @staticmethod   
    def resin(state: Status) -> list[Order]:

        current_price = state.maxamt_midprc

        orders = []
        orders.extend(Strategy.arb(state=state, fair_price=current_price))
        orders.extend(Strategy.mm_ou(state=state, fair_price=current_price, gamma=1e-9, order_amount=50))

        return orders
    
    @staticmethod
    def kelp(state: Status) -> list[Order]:

        current_price = state.maxamt_midprc

        orders = []
        orders.extend(Strategy.arb(state=state, fair_price=current_price))
        orders.extend(Strategy.mm_glft(state=state, fair_price=current_price, mu = 1.2484084052394708e-07, sigma = 0.0001199636554242691, gamma=1e-9, order_amount=50))

        return orders
    
    def compute_ema(prices: list[float], period: int) -> float:
        alpha = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema

    # @staticmethod
    # def squink(state: Status) -> list[Order]:
    #     required_points = EMA_PERIOD + 1
    #     hist = state.hist_mid_prc(required_points)
    #     if len(hist) < required_points:
    #         return []

    #     slow_ema = Trade.compute_ema(hist, EMA_PERIOD)
    #     fast_period = max(2, EMA_PERIOD // 2)
    #     fast_ema = Trade.compute_ema(hist, fast_period)
        
    #     macd = fast_ema - slow_ema

    #     ema_previous = Trade.compute_ema(hist[:-1], EMA_PERIOD)
    #     ema_current = Trade.compute_ema(hist[1:], EMA_PERIOD)
    #     derivative = ema_current - ema_previous

    #     sigma = np.std(hist)
    #     current = state.mid
    #     z = (current - slow_ema) / sigma if sigma > 0 else 0

    #     min_deviation = 1.0
    #     stop_loss_threshold = 2.5
    #     max_position_threshold = 20

    #     orders = []

    #     if abs(z) >= stop_loss_threshold or abs(state.rt_position) > max_position_threshold:
    #         minimal_qty = 1
    #         logger.print("Stop loss condition met: z =", z, "or high position =", state.rt_position)
    #         logger.print("Trading minimal volume:", minimal_qty)
    #         if z > 0:
    #             return [Order(state.product, int(current), -minimal_qty)]
    #         else:
    #             return [Order(state.product, int(current), minimal_qty)]
        
    #     if abs(z) < min_deviation:
    #         return []

    #     risk_factor = 1 / (1 + abs(derivative))
    #     base_qty = max(1, int(MAX_BASE_QTY * risk_factor))
        
    #     macd_scale = 1.0 + min(0.5, abs(macd))
    #     base_qty = int(base_qty * macd_scale)
        
    #     order_scale = min(1.0, abs(z) / 2)
    #     order_qty = max(1, int(base_qty * order_scale))
        
    #     print("SQUID_STRAT: z =", z, "derivative =", derivative, "MACD =", macd, 
    #         "base_qty =", base_qty, "order_qty =", order_qty)
        
    #     if current < slow_ema:
    #         orders.append(Order(state.product, int(current), order_qty))
    #     elif current > slow_ema:
    #         orders.append(Order(state.product, int(current), -order_qty))
        
    #     return orders
    @staticmethod
    def ema_mean_reversion(squink: Status, alpha=0.15, threshold=14):
        orders = []
        squink_prc = squink.mid  # This is a float

        # Ensure squink has an attribute for historical prices.
        if not hasattr(squink, 'price_history'):
            squink.price_history = []
            
        # Append the current price to the history.
        squink.price_history.append(squink_prc)
        
        # Only compute the EMA if we have enough history (e.g., at least 10 data points)
        if len(squink.price_history) < 100:
            return orders  # or you can decide to simply return no orders
        
        # Convert the price history to a Pandas Series
        price_series = pd.Series(squink.price_history)
        
        # Compute the EMA using Pandas' ewm method
        ema = price_series.ewm(alpha=alpha, adjust=False).mean().iloc[-1]

        if squink_prc > ema + threshold:
            orders.append(Order(squink.product, int(squink.best_bid), -int(squink.possible_sell_amt)))
        elif squink_prc < ema - threshold:
            orders.append(Order(squink.product, int(squink.best_ask), int(squink.possible_buy_amt)))
        return orders
    
    @staticmethod
    def jams(state: Status) -> list[Order]:

        current_price = state.maxamt_midprc

        orders = []
        orders.extend(Strategy.arb(state=state, fair_price=current_price))
        orders.extend(Strategy.mm_glft(state=state, fair_price=current_price, mu = -7.60706813499185e-07, sigma = 7.890239872766339e-05, gamma=1e-9, order_amount=50))

        return orders
    
    @staticmethod
    def djmb_crs_pair(state_djembes: Status, state_croiss: Status) -> List[Order]:
        return Strategy.pair_trade(croissant=state_croiss, djembes=state_djembes)
    
    @staticmethod
    def basket_1(basket: Status, jam: Status, djembes: Status, croissant: Status) -> list[Order]:

        orders = []
        orders.extend(Strategy.index_arb(basket, jam, djembes, croissant, theta = 3.65410486e-07, threshold=69, jam_m = 3, croiss_m = 6, djembe_m = 1))

        return orders

    @staticmethod
    def basket_2(basket: Status, jam: Status, djembes: Status, croissant: Status) -> list[Order]:

        orders = []
        orders.extend(Strategy.index_arb(basket, jam, djembes, croissant, theta = 1.33444695e+01, threshold=47, jam_m = 2, croiss_m = 4, djembe_m = 0))

        return orders
    
    @staticmethod
    def convert(state: Status) -> int:
        return Strategy.convert(state=state)
    
    @staticmethod
    def croissant_ema(state: TradingState) -> list[Order]:
        # Access the class attribute instead of a global variable.
        status = Status(CROISSANTS)
        Status.cls_update(state)
        current_price = status.mid

        history = Trade.mid_price_history[CROISSANTS]
        history.append(current_price)
        if len(history) > EMA_PERIOD:
            history.pop(0)
            
        if len(history) < EMA_PERIOD:
            return []
        
        alpha = 2 / (EMA_PERIOD + 1)
        ema = history[0]
        for price in history[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        std = np.std(history)
        z_score = (current_price - ema) / std if std > 0 else 0
        
        logger.print("Croissants EMA Strategy:", "Current Price =", current_price, "EMA =", ema, "Std =", std, "z =", z_score)
        
        orders = []
        if z_score < -2.8:
            qty = status.possible_buy_amt
            orders.append(Order(CROISSANTS, int(current_price), qty))
            logger.print("EMA Signal: BUY CROISSANTS", "Price =", current_price, "Quantity =", qty)
        elif z_score > 2.8:
            qty = status.possible_sell_amt
            orders.append(Order(CROISSANTS, int(current_price), -qty))
            logger.print("EMA Signal: SELL CROISSANTS", "Price =", current_price, "Quantity =", qty)
        else:
            logger.print("EMA Signal: No action (z-score within threshold).")
            
        return orders
    
    @staticmethod
    def voucher_trade(voucher_state: 'Status', strike: float, round_number: int, current_timestamp: int) -> List[Order]:
        return Strategy.voucher_trade(voucher_state, strike, round_number, current_timestamp)
    

    @staticmethod
    def volcanic_rock(state: Status) -> list[Order]:
        current_price = state.mid
        if state.position < 0 and state.position < -int(state.position_limit * 0.5):
            logger.print("Stop-loss active on VOLCANIC_ROCK: reducing order size")
            order_qty = 1
            return [Order(state.product, int(current_price), -order_qty)]
        orders = []
        orders.extend(Strategy.arb(state=state, fair_price=current_price))
        orders.extend(Strategy.mm_ou(state=state, fair_price=current_price, gamma=1e-9, order_amount=50))
        return orders

    
class Trader:
    state_resin = Status("RAINFOREST_RESIN")
    state_kelp = Status("KELP")
    state_squink = Status("SQUID_INK")
    state_croiss = Status("CROISSANTS")
    state_jam = Status("JAMS")
    state_djembes = Status("DJEMBES")
    state_picnic1 = Status("PICNIC_BASKET1")
    state_picnic2 = Status("PICNIC_BASKET2")
    state_voucher_9500 = Status("VOLCANIC_ROCK_VOUCHER_9500")
    state_voucher_9750 = Status("VOLCANIC_ROCK_VOUCHER_9750")
    state_voucher_10000 = Status("VOLCANIC_ROCK_VOUCHER_10000")
    state_voucher_10250 = Status("VOLCANIC_ROCK_VOUCHER_10250")
    state_voucher_10500 = Status("VOLCANIC_ROCK_VOUCHER_10500")
    state_volcanic_rock = Status("VOLCANIC_ROCK")
    
    last_vol_coeffs = None
    voucher_deltas = {}

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        Status.cls_update(state)
        round_number = 3
        current_timestamp = state.timestamp

        result = {}

        # Round 1 orders:
        result["RAINFOREST_RESIN"] = Trade.resin(self.state_resin)
        result["KELP"] = Trade.kelp(self.state_kelp)
        result["SQUID_INK"] = Trade.ema_mean_reversion(self.state_squink)

        # Round 2 orders:
        result["PICNIC_BASKET1"] = Trade.basket_1(self.state_picnic1, self.state_jam, self.state_djembes, self.state_croiss)
        result["JAMS"] = Trade.jams(self.state_jam)
        result["PICNIC_BASKET2"] = Trade.basket_2(self.state_picnic2, self.state_jam, self.state_djembes, self.state_croiss)

        # --- Volcanic Strategy (Round 3) ---
        # Define voucher symbols, states, and strikes
        voucher_symbols = [
            "VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750",
            "VOLCANIC_ROCK_VOUCHER_10000", "VOLCANIC_ROCK_VOUCHER_10250",
            "VOLCANIC_ROCK_VOUCHER_10500"
        ]
        
        try:
            voucher_states = {
                "VOLCANIC_ROCK_VOUCHER_9500": self.state_voucher_9500,
                "VOLCANIC_ROCK_VOUCHER_9750": self.state_voucher_9750,
                "VOLCANIC_ROCK_VOUCHER_10000": self.state_voucher_10000,
                "VOLCANIC_ROCK_VOUCHER_10250": self.state_voucher_10250,
                "VOLCANIC_ROCK_VOUCHER_10500": self.state_voucher_10500,
            }
            # Also ensure the underlying state exists
            if not hasattr(self, 'state_volcanic_rock'):
                 raise AttributeError("state_volcanic_rock not defined in Trader")

        except AttributeError as e:
             logger.print(f"CRITICAL ERROR: Missing state attribute in Trader class: {e}")
             logger.flush(state, result, 0, "AttributeError") # Log error and exit run
             return result, 0, "AttributeError"
        
        
        strikes = {
            "VOLCANIC_ROCK_VOUCHER_9500": 9500, "VOLCANIC_ROCK_VOUCHER_9750": 9750,
            "VOLCANIC_ROCK_VOUCHER_10000": 10000, "VOLCANIC_ROCK_VOUCHER_10250": 10250,
            "VOLCANIC_ROCK_VOUCHER_10500": 10500
        }

        # 1. Get Inputs
        try:
            # CRITICAL: Use VOLCANIC_ROCK price as spot (St in hint)
            spot_volcanic = self.state_volcanic_rock.mid
            TTE = compute_time_to_expiry(round_number, current_timestamp)
            r = 0.0 # Risk-free rate
            # Ensure TTE is valid
            if TTE <= 1e-9: # Avoid division by zero or tiny TTE issues
                 raise ValueError(f"Time to expiry is too small or zero: {TTE}")
        except Exception as e:
            logger.print(f"Error getting Volcanic inputs or invalid TTE: {e}. Skipping Volcanic strategy.")
            logger.flush(state, result, 1, "ErrorState")
            return result, 1, "ErrorState"

        # 2. Calculate m_t and actual_v_t (Implied Vol) for all vouchers
        valid_m_values = []
        valid_v_values = []
        current_ivs = {}
        current_moneyness = {}

        for symbol in voucher_symbols:
            voucher_state = voucher_states[symbol]
            try:
                # Vt in hint = voucher market price
                market_price = voucher_state.mid
                K = strikes[symbol]

                if market_price <= 0 or spot_volcanic <= 0:
                    logger.print(f"Skipping IV/M calc for {symbol}: Invalid prices (S={spot_volcanic:.2f}, V={market_price:.2f})")
                    current_ivs[symbol] = np.nan
                    current_moneyness[symbol] = np.nan
                    continue

                # Calculate Implied Volatility (using correct spot)
                # v_t = BlackScholes ImpliedVol(St, Vt, K, TTE)
                actual_v_t = implied_volatility(market_price, spot_volcanic, K, round_number, current_timestamp, r)
                current_ivs[symbol] = actual_v_t

                # Calculate Moneyness
                # m_t = log(K/St)/ sqrt(TTE)
                m_t = compute_m_t(spot_volcanic, K, TTE)
                current_moneyness[symbol] = m_t

                # Filter unreliable values before fitting
                # (Adjust bounds as needed based on testing)
                if not np.isnan(actual_v_t) and not np.isnan(m_t) and 0.05 < actual_v_t < 1.5:
                     valid_m_values.append(m_t)
                     valid_v_values.append(actual_v_t)
                     logger.print(f"Valid point for fit: {symbol} m_t={m_t:.4f}, v_t={actual_v_t:.4f}")
                else:
                    logger.print(f"Filtered out IV/M for {symbol}: m_t={m_t:.4f}, actual_v_t={actual_v_t:.4f}")

            except Exception as e:
                logger.print(f"Error calculating IV/M for {symbol}: {e}")
                current_ivs[symbol] = np.nan
                current_moneyness[symbol] = np.nan

        # 3. Fit Parabolic Curve (Requires numpy, which is allowed)
        fitted_coeffs = None
        if len(valid_m_values) >= 3: # Need 3+ points for quadratic fit
            try:
                # v = a*m^2 + b*m + c  <- polyfit returns [a, b, c]
                fitted_coeffs = np.polyfit(valid_m_values, valid_v_values, 2)
                self.last_vol_coeffs = fitted_coeffs # Store for traderData
                base_IV = math.log(fitted_coeffs[2]) # c term = fitted v_t(m_t=0)
                logger.print(f"Fit Parabola Coeffs (a,b,c): {fitted_coeffs}, Base IV (c) = {base_IV:.4f}")
            except Exception as e:
                logger.print(f"Error fitting parabola: {e}")
                fitted_coeffs = None
        else:
            logger.print(f"Not enough valid points ({len(valid_m_values)}) to fit parabola.")
            fitted_coeffs = None

        # 4. Calculate Fitted Prices and Deltas (using fitted_v) & Generate Voucher Orders
        net_voucher_delta_change = 0.0
        threshold = 1.5 # Example threshold for mispricing signal (tune this)

        for symbol in voucher_symbols:
            symbol_orders = []
            voucher_state = voucher_states[symbol]
            try:
                market_price = voucher_state.mid # Vt
                K = strikes[symbol]
                m_t = current_moneyness.get(symbol, np.nan)
                delta = np.nan # Initialize delta

                if fitted_coeffs is not None and not np.isnan(m_t):
                    # Evaluate fitted volatility: fitted_v = a*m^2 + b*m + c
                    fitted_v = evaluate_parabolic(m_t, fitted_coeffs)
                    # Clamp fitted_v to prevent extreme values in BS
                    fitted_v = max(0.01, min(1.5, fitted_v)) # Adjust bounds as needed

                    # Calculate theoretical price using the fitted volatility
                    fitted_price = bs_coupon_price(spot_volcanic, K, TTE, r, fitted_v)

                    # Calculate Delta using fitted_v and CUSTOM norm_cdf
                    if fitted_v * math.sqrt(TTE) > 1e-9: # Avoid division by zero
                         sqrt_T = math.sqrt(TTE)
                         d1 = (math.log(spot_volcanic / K) + (r + 0.5 * fitted_v**2) * TTE) / (fitted_v * sqrt_T)
                         delta = norm_cdf(d1) # USE YOUR CUSTOM norm_cdf
                    else:
                         delta = 0.0 if spot_volcanic <= K else 1.0 # Intrinsic delta
                    self.voucher_deltas[symbol] = delta # Store delta

                    logger.print(f"{symbol}: M={m_t:.2f}, ActualV={current_ivs.get(symbol, np.nan):.4f}, FittedV={fitted_v:.4f}, MarketP={market_price:.2f}, FittedP={fitted_price:.2f}, Delta={delta:.3f}")

                    # Generate Orders based on mispricing vs fitted curve
                    if market_price < fitted_price - threshold:
                        qty = voucher_state.possible_buy_amt
                        if qty > 0:
                            # Aggressive buy: offer slightly above market mid or at fitted price? Test this.
                            order_price = int(math.ceil(market_price + 0.5)) # Example: round up mid
                            order = Order(symbol, order_price, qty)
                            symbol_orders.append(order)
                            net_voucher_delta_change += delta * qty # Accumulate delta change
                            logger.print(f"-> VOUCHER BUY {symbol} @ {order.price} x {qty}")
                    elif market_price > fitted_price + threshold:
                        qty = voucher_state.possible_sell_amt
                        if qty > 0:
                            # Aggressive sell: offer slightly below market mid
                            order_price = int(math.floor(market_price - 0.5)) # Example: round down mid
                            order = Order(symbol, order_price, -qty)
                            symbol_orders.append(order)
                            # Delta change is delta * (-qty) = -delta * qty
                            net_voucher_delta_change -= delta * qty # Accumulate delta change
                            logger.print(f"-> VOUCHER SELL {symbol} @ {order.price} x {-qty}")
                else:
                    # No fit or invalid m_t, clear stored delta
                    self.voucher_deltas.pop(symbol, None)
                    logger.print(f"{symbol}: Skipping trade signal generation (no valid fit or inputs).")

                # Add orders for this symbol to the main result dictionary
                if symbol_orders:
                    if symbol not in result: result[symbol] = []
                    result[symbol].extend(symbol_orders)

            except Exception as e:
                logger.print(f"Error processing trade signal for {symbol}: {e}")
                self.voucher_deltas.pop(symbol, None) # Clear delta on error


        # 5. Calculate Net Portfolio Delta and Hedge Order
        current_portfolio_delta = 0.0
        for symbol in voucher_symbols:
            position = voucher_states[symbol].position # Current position
            delta = self.voucher_deltas.get(symbol, 0.0) # Use stored delta
            current_portfolio_delta += position * delta

        # Total delta after planned trades = current delta + delta from new trades
        total_target_delta = current_portfolio_delta + net_voucher_delta_change

        # Target position in underlying to be delta neutral
        target_hedge_position = -round(total_target_delta)

        # Calculate hedge order quantity needed
        current_hedge_position = self.state_volcanic_rock.position
        hedge_order_qty = target_hedge_position - current_hedge_position

        # Apply VOLCANIC_ROCK position limits (ensure Status class property is correct)
        hedge_limit = self.state_volcanic_rock.position_limit
        if hedge_order_qty > 0: # Buying hedge
            hedge_order_qty = min(hedge_order_qty, hedge_limit - current_hedge_position)
        elif hedge_order_qty < 0: # Selling hedge
            hedge_order_qty = max(hedge_order_qty, -hedge_limit - current_hedge_position)

        logger.print(f"Delta Hedge: CurrentVoucherDelta={current_portfolio_delta:.2f}, NewTradeDelta={net_voucher_delta_change:.2f}, TargetNetDelta={total_target_delta:.2f}")
        logger.print(f"Hedge Calc: TargetHedgePos={target_hedge_position}, CurrentHedgePos={current_hedge_position}, OrderQty={hedge_order_qty}")

        # Create hedge order for VOLCANIC_ROCK if needed
        if abs(hedge_order_qty) > 0:
             # Use underlying's price for hedge order - try to cross spread slightly
            best_bid_vr = self.state_volcanic_rock.best_bid
            best_ask_vr = self.state_volcanic_rock.best_ask
            if hedge_order_qty > 0: # Buying hedge
                 hedge_price = best_ask_vr # Take best ask
            else: # Selling hedge
                 hedge_price = best_bid_vr # Take best bid

            # Ensure valid price exists (market might be wide or empty)
            if best_bid_vr < best_ask_vr : # Basic check for valid market
                hedge_order = Order("VOLCANIC_ROCK", hedge_price, hedge_order_qty)
                logger.print(f"-> HEDGE ORDER: {hedge_order.symbol} @ {hedge_order.price} x {hedge_order.quantity}")
                if "VOLCANIC_ROCK" not in result: result["VOLCANIC_ROCK"] = []
                result["VOLCANIC_ROCK"].append(hedge_order)
            else:
                logger.print(f"Cannot place hedge order: Invalid market BBO ({best_bid_vr} / {best_ask_vr})")
        else:
             logger.print("No hedge order needed.")

        # Optional: If no hedge order was placed, consider adding back the original MM/Arb logic
        # if "VOLCANIC_ROCK" not in result:
        #     result["VOLCANIC_ROCK"] = Trade.volcanic_rock(self.state_volcanic_rock)


        # --- Final Steps ---
        conversions = 0 # Set conversions if needed by other strategies

        # Serialize traderData - CHECK IF jsonpickle IS ALLOWED
        try:
            # If jsonpickle is allowed:
            # import jsonpickle
            serializable_coeffs = self.last_vol_coeffs # Keep as numpy array
            trader_data_dict = {"last_vol_coeffs": serializable_coeffs, "voucher_deltas": self.voucher_deltas}
            traderData = jsonpickle.encode(trader_data_dict)

            # # If jsonpickle is NOT allowed:
            # import json
            # serializable_coeffs = None
            # if self.last_vol_coeffs is not None:
            #      serializable_coeffs = self.last_vol_coeffs.tolist() # Convert numpy array to list
            # trader_data_dict = {"last_vol_coeffs": serializable_coeffs, "voucher_deltas": self.voucher_deltas}
            # traderData = json.dumps(trader_data_dict, separators=(",", ":"))

        except Exception as e:
            logger.print(f"Error serializing traderData: {e}")
            traderData = "" # Fallback to empty string


        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData