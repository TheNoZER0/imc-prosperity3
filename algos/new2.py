from typing import Any, List, Dict, Tuple
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
MACARON_PRODUCT = "MAGNIFICENT_MACARONS"
MACARON_POSITION_LIMIT = 75
MACARON_CSI_THRESHOLD = 40
MACARON_CSI_SUSTAINED_TICKS = 15
MACARON_STORAGE_COST_PER_TICK = 0.1 / 100 # 0.001 per tick (less impactful in this model)
MACARON_CONVERSION_LIMIT = 10
MACARON_DEFAULT_SPREAD = 10 # Base spread, will be adapted
MACARON_MM_ORDER_SIZE = 10
# Adaptive Edge Params (Tune these heavily)
MACARON_ADAPT_EDGE_STEP = 0.1
MACARON_ADAPT_MIN_EDGE = 0.5
MACARON_ADAPT_MAX_EDGE = 4.0 # Allow wider max edge
MACARON_ADAPT_POS_HIST_LEN = 5
MACARON_ADAPT_VOL_THRESH_UP = 25 # Pos Std Dev threshold to increase edge
MACARON_ADAPT_VOL_THRESH_DOWN = 10 # Pos Std Dev threshold to decrease edge
MACARON_ADAPT_POS_THRESH_DOWN = 10 # Max position abs value to allow edge decrease
# Take Logic Probability Factor (similar to Orchid make_probability) - Scales edge for taking
MACARON_TAKE_EDGE_PROB_FACTOR = 0.6 # Tune: Aggressiveness of taking vs implied price



def compute_time_to_expiry(round_number: int, current_timestamp: int) -> float:
    total_round_units = 7_000_000
    day_units = 1_000_000
    remaining_units = total_round_units - (((round_number - 1) * day_units) + current_timestamp)
    return remaining_units

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
    Calculate the Black-Scholes price for a voucher (European call option)
    using statistics.NormalDist
    """

    sqrt_T = math.sqrt(time_to_expiry)
    d1 = (math.log(spot / strike) + (0.5*volatility**2)*time_to_expiry) / (volatility * sqrt_T)
    d2 = d1 - volatility * sqrt_T
    price = spot * NormalDist().cdf(d1) - strike*NormalDist().cdf(d2)
    return price
    
def calculate_delta(spot: float, strike: float, time_to_expiry: float,
                    risk_free_rate: float, volatility: float) -> float:
    """
    Calculate the Black-Scholes delta for a voucher (European call option)
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
                       current_timestamp: int, r: float = 0.0,
                       tol: float = 1e-7,
                       max_iter: int = 1000
                       ) -> float: # Return type is float (np.nan is a float)
    """Calculates implied volatility using bisection method. Returns np.nan on failure."""
    TTE = compute_time_to_expiry(round_number, current_timestamp)

    def objective(vol):
        calc_price = bs_coupon_price(spot, strike, TTE, r, vol)
        return calc_price - market_price

    # --- Bisection Search ---
    lower_bound_vol = 1e-7
    upper_bound_vol = 1e-5
    vol_low = lower_bound_vol
    vol_high = upper_bound_vol
    obj_low = objective(vol_low)
    obj_high = objective(vol_high)

    if obj_low * obj_high >= 0:
        if abs(obj_low) < abs(obj_high) : return vol_low
        elif abs(obj_high) < abs(obj_low) : return vol_high
        else: return np.nan

    for _ in range(max_iter):
        vol_mid = (vol_low + vol_high) / 2.0
        obj_mid = objective(vol_mid)

        if abs(obj_mid) < tol:
            return vol_mid

        if obj_low * obj_mid < 0:
            vol_high = vol_mid
            obj_high = obj_mid
        else:
            vol_low = vol_mid
            obj_low = obj_mid

        if abs(vol_low - vol_high) < tol:
             return (vol_low + vol_high) / 2.0

    return (vol_low + vol_high) / 2.0

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
    # Here we use the recent price volatility (sigma_hist) multiplied by vol_multiplier as a proxy for volatility.
    volatility = sigma_hist * vol_multiplier
    TTE = compute_time_to_expiry(round_number, current_timestamp)
    
    # Compute option parameters.
    d1 = (math.log(S / strike) + (r + 0.5 * volatility**2) * TTE) / (volatility * math.sqrt(TTE))
    d2 = d1 - volatility * math.sqrt(TTE)
    fair_value = bs_coupon_price(S, strike, TTE, r, volatility)
    
    # Record some history (you already have these for plotting/curve fit).
    current_m = compute_m_t(S, strike, TTE)
    current_v = implied_volatility(S, S, strike, round_number, current_timestamp)
    voucher_m_history.append(current_m)
    voucher_v_history.append(current_v)
    
    if len(voucher_m_history) >= 20:
        m_array = np.array(voucher_m_history[-20:])
        v_array = np.array(voucher_v_history[-20:])
        coeffs = fit_parabolic(m_array, v_array)
        base_IV = math.log(coeffs[2])  # Base IV is the constant term at m=0 (logged).
        #logger.print(f"Fitted Parabola Coeffs: {coeffs}, Base IV = {base_IV:.2f}")
    else:
        base_IV = None

    # *** INSERT DELTA COMPUTATION SNIPPET HERE ***
    # Compute delta using the IV we computed (or use current_v as the IV proxy)
    if not np.isnan(current_v) and current_v > 1e-6:
        delta = calculate_delta(S, strike, TTE, r, current_v)
    else:
        delta = 1.0 if S > strike else 0.0

    # Persist the computed delta in Trader.voucher_deltas using a consistent key format.
    Trader.voucher_deltas[str(int(strike))] = delta
    #logger.print(f"[Delta Info] For strike {strike}: Delta = {delta:.2f}")
    # *** END OF DELTA COMPUTATION SNIPPET ***

    # Now, use a dynamic buffer based on volatility to set target buy/sell thresholds:
    buffer = base_buffer * volatility
    target_buy = fair_value - buffer
    target_sell = fair_value + buffer

    #logger.print(f"[Voucher Trade] {voucher_state.product}: S = {S:.2f}, Fair Value = {fair_value:.2f}, "
                 #f"Target Buy = {target_buy:.2f}, Target Sell = {target_sell:.2f}, Delta = {norm_cdf(d1):.2f}")

    orders = []
    if S < target_buy:
        qty = voucher_state.possible_buy_amt
        if qty > 0:
            orders.append(Order(voucher_state.product, int(S), qty))
            hedge_qty = int(hedge_scale * norm_cdf(d1) * qty)
            if hedge_qty > 0:
                orders.append(Order("VOLCANIC_ROCK", int(voucher_state.mid), -hedge_qty))
            #logger.print(f"Voucher BUY: {voucher_state.product} qty = {qty}, hedge_qty = {hedge_qty}")
    elif S > target_sell:
        qty = voucher_state.possible_sell_amt
        if qty > 0:
            orders.append(Order(voucher_state.product, int(S), -qty))
            hedge_qty = int(hedge_scale * norm_cdf(d1) * qty)
            if hedge_qty > 0:
                orders.append(Order("VOLCANIC_ROCK", int(voucher_state.mid), hedge_qty))
            #logger.print(f"Voucher SELL: {voucher_state.product} qty = {qty}, hedge_qty = {hedge_qty}")
    
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
                observation.sunlightIndex,
                observation.sugarPrice,
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
        "MAGNIFICENT_MACARONS": 75,
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
        

# --- Macaron Strategy State ---
class MacaronStrategyState:
    def __init__(self):
        self.ticks_below_csi = 0
        self.current_edge = MACARON_DEFAULT_SPREAD
        self.position_history: List[int] = []
        self.current_pnl = 0  # pnl starts at 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticks_below_csi": self.ticks_below_csi,
            "current_edge": self.current_edge,
            "position_history": self.position_history  # already a list
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MacaronStrategyState":
        instance = cls()
        instance.ticks_below_csi = data.get("ticks_below_csi", 0)
        instance.current_edge = data.get("current_edge", MACARON_DEFAULT_SPREAD)
        loaded_history = data.get("position_history", [])
        instance.position_history = list(loaded_history) if isinstance(loaded_history, (list, np.ndarray)) else []
        if len(instance.position_history) > MACARON_ADAPT_POS_HIST_LEN:
            instance.position_history = instance.position_history[-MACARON_ADAPT_POS_HIST_LEN:]
        return instance

# --- Macaron Strategy ---
class MacaronStrategy:
    def __init__(self, logger: Any):
        self.logger = logger
        self.strategy_state = MacaronStrategyState()

    # --- State Persistence ---
    def update_state_from_traderdata(self, traderData: Dict[str, Any]):
        if MACARON_PRODUCT in traderData:
            self.strategy_state = MacaronStrategyState.from_dict(traderData[MACARON_PRODUCT])
        else:
            self.strategy_state = MacaronStrategyState()

    def save_state_to_traderdata(self, traderData: Dict[str, Any]):
        traderData[MACARON_PRODUCT] = self.strategy_state.to_dict()

    # --- CSI and Mode ---
    def _update_csi_state(self, sunlight_index: float):
        if sunlight_index < MACARON_CSI_THRESHOLD:
            self.strategy_state.ticks_below_csi += 1
        else:
            self.strategy_state.ticks_below_csi = 0

    def _get_trading_mode(self, timestamp: int) -> str:
        # Here's a simple mode selection: if near round end, trigger flattening; else if sustained low CSI, go PANIC; otherwise NORMAL.
        if timestamp >= 995000:
            return "FLATTENING_URGENT"
        elif self.strategy_state.ticks_below_csi >= MACARON_CSI_SUSTAINED_TICKS:
            return "PANIC"
        else:
            return "NORMAL"

    # --- Price Levels ---
    def _calculate_pristine_levels(self, obs: ConversionObservation) -> Tuple[float, float]:
        pristine_buy = obs.askPrice + obs.transportFees + obs.importTariff
        pristine_sell = obs.bidPrice - obs.transportFees - obs.exportTariff
        return pristine_buy, pristine_sell

    # --- Edge Adaptation ---
    def _adapt_edge(self, current_position: int, timestamp: int, mode: str) -> float:
        # Update rolling history
        self.strategy_state.position_history.append(current_position)
        if len(self.strategy_state.position_history) > MACARON_ADAPT_POS_HIST_LEN:
            self.strategy_state.position_history.pop(0)
        # Warmup period: not enough data or very early tick
        if len(self.strategy_state.position_history) < MACARON_ADAPT_POS_HIST_LEN or timestamp < 1000:
            return self.strategy_state.current_edge

        pos_std = np.std(self.strategy_state.position_history)
        current_edge = self.strategy_state.current_edge
        changed = False

        if pos_std > MACARON_ADAPT_VOL_THRESH_UP:
            current_edge = min(MACARON_ADAPT_MAX_EDGE, current_edge + MACARON_ADAPT_EDGE_STEP)
            self.logger.print(f"Edge INC to {current_edge:.2f} (Std: {pos_std:.2f})")
            changed = True
        elif pos_std < MACARON_ADAPT_VOL_THRESH_DOWN and abs(current_position) < MACARON_ADAPT_POS_THRESH_DOWN:
            current_edge = max(MACARON_ADAPT_MIN_EDGE, current_edge - MACARON_ADAPT_EDGE_STEP)
            self.logger.print(f"Edge DEC to {current_edge:.2f} (Std: {pos_std:.2f}, Pos: {current_position})")
            changed = True

        # Mode adjustments
        if mode == "PANIC":
            current_edge = max(current_edge, MACARON_DEFAULT_SPREAD * 1.2)
            self.logger.print(f"Panic mode: Edge set to {current_edge:.2f}")
        elif mode.startswith("FLATTENING"):
            # Optionally tighten edge in flattening mode:
            current_edge = max(MACARON_ADAPT_MIN_EDGE, current_edge * 0.9)
            self.logger.print(f"Flattening mode: Edge adjusted to {current_edge:.2f}")

        if changed:
            self.strategy_state.position_history = []  # Reset history after change

        self.strategy_state.current_edge = current_edge
        return current_edge

    # --- Conversion (Flattening) ---
    def _request_conversion(self, current_position: int) -> int:
        if current_position == 0:
            return 0
        conv = -current_position  # conversion to zero out the position
        return max(-MACARON_CONVERSION_LIMIT, min(MACARON_CONVERSION_LIMIT, conv))

    # --- Order Generation ---
    def _take_local_orders(self, state: Status, pristine_buy: float, pristine_sell: float,
                           edge: float, prob_factor: float) -> List[Order]:
        orders = []
        effective_edge = edge * prob_factor

        # Buy takes: if an ask is below (pristine_sell - effective_edge)
        buy_threshold = pristine_sell - effective_edge
        for price, avail in sorted(state.asks):
            if price < buy_threshold:
                qty = min(-avail, state.possible_buy_amt)
                if qty > 0:
                    orders.append(Order(MACARON_PRODUCT, price, qty))
                    state.rt_position_update(state.rt_position + qty)
                    self.logger.print(f"Take Buy: {qty}@{price} (Thresh {buy_threshold:.2f})")
                    if state.rt_position >= MACARON_POSITION_LIMIT:
                        break
            else:
                break

        # Sell takes: if a bid is above (pristine_buy + effective_edge)
        sell_threshold = pristine_buy + effective_edge
        for price, avail in sorted(state.bids, reverse=True):
            if price > sell_threshold:
                qty = min(avail, state.possible_sell_amt)
                if qty > 0:
                    orders.append(Order(MACARON_PRODUCT, price, -qty))
                    state.rt_position_update(state.rt_position - qty)
                    self.logger.print(f"Take Sell: {-qty}@{price} (Thresh {sell_threshold:.2f})")
                    if state.rt_position <= -MACARON_POSITION_LIMIT:
                        break
            else:
                break

        return orders

    def _make_local_orders(self, state: Status, pristine_buy: float, pristine_sell: float,
                           edge: float) -> List[Order]:
        orders = []
        # Determine target bid and ask prices
        target_bid = int(round(pristine_sell - edge))
        target_ask = int(round(pristine_buy + edge))
        if target_bid >= target_ask:
            mid = (pristine_buy + pristine_sell) / 2
            target_bid = int(mid - 1)
            target_ask = int(mid + 1)
        bid_qty = min(MACARON_MM_ORDER_SIZE, state.possible_buy_amt)
        if bid_qty > 0:
            orders.append(Order(MACARON_PRODUCT, target_bid, bid_qty))
            self.logger.print(f"Make Bid: {bid_qty}@{target_bid}")
        ask_qty = min(MACARON_MM_ORDER_SIZE, state.possible_sell_amt)
        if ask_qty > 0:
            orders.append(Order(MACARON_PRODUCT, target_ask, -ask_qty))
            self.logger.print(f"Make Ask: {-ask_qty}@{target_ask}")
        return orders

    # --- Main Run Method ---
    def run(self, state: Status, timestamp: int) -> Tuple[List[Order], int]:
        all_orders: List[Order] = []
        # 1. Get conversion observation
        obs = state._state.observations.conversionObservations.get(MACARON_PRODUCT)
        if not obs:
            self.logger.print(f"WARN ({timestamp}): No ConversionObservation for {MACARON_PRODUCT}")
            return [], 0

        # 2. Update CSI state and determine mode
        current_position = state.position
        self._update_csi_state(obs.sunlightIndex)
        mode = self._get_trading_mode(timestamp)
        state.rt_position_update(current_position)

        # 3. Request flattening conversion only in flattening modes
        conv_req = 0
        if mode.startswith("FLATTENING"):
            conv_req = self._request_conversion(current_position)
            state.rt_position_update(current_position + conv_req)
            self.logger.print(f"Flattening: Request conversion {conv_req}")
        else:
            self.logger.print("Normal mode: No conversion requested")

        self.logger.print(f"Initial Pos: {current_position}, RT Pos: {state.rt_position}")
        # 4. Calculate target/pristine levels and adaptive edge
        pristine_buy, pristine_sell = self._calculate_pristine_levels(obs)
        edge_val = self._adapt_edge(current_position, timestamp, mode)
        self.logger.print(f"Mode: {mode}, Edge: {edge_val:.2f}, ImpBuy: {pristine_buy:.2f}, ImpSell: {pristine_sell:.2f}")

        # 5. Generate aggressive orders ('take')
        take_orders = self._take_local_orders(state, pristine_buy, pristine_sell, edge_val, MACARON_TAKE_EDGE_PROB_FACTOR)
        all_orders.extend(take_orders)

        # 6. Generate resting orders ('make') if not flattening
        if mode == "NORMAL":
            make_orders = self._make_local_orders(state, pristine_buy, pristine_sell, edge_val)
            all_orders.extend(make_orders)

        # 7. Safety net: if in flattening mode, ensure position is 0
        if mode.startswith("FLATTENING") and state.rt_position != 0:
            self.logger.print(f"Safety net flattening: RT Pos {state.rt_position}")
            if state.rt_position > 0:
                for price, vol in sorted(state.bids, reverse=True):
                    qty = min(state.rt_position, vol)
                    if qty > 0:
                        all_orders.append(Order(MACARON_PRODUCT, price, -qty))
                        state.rt_position_update(state.rt_position - qty)
                        if state.rt_position == 0:
                            break
            elif state.rt_position < 0:
                for price, vol in sorted(state.asks):
                    qty = min(-state.rt_position, -vol)
                    if qty > 0:
                        all_orders.append(Order(MACARON_PRODUCT, price, qty))
                        state.rt_position_update(state.rt_position + qty)
                        if state.rt_position == 0:
                            break

        return all_orders, conv_req

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
EMA_PERIOD = 20
PARITY_MARGIN = 0.5      # Window period for the EMA calculation.
class Trade:
    mid_price_history = {CROISSANTS: []}
    macarons_history = {
        "mid": [],
        "sugar": [],
        "sunlight": [],
        "transportFees": [],
        "exportTariff": [],
        "importTariff": []
    }
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

    @staticmethod
    # alpha for squink is 0.1, thresh is 8
    def ema_mean_reversion(squink: Status, alpha, threshold):
        orders = []
        squink_prc = squink.mid  # This is a float

        # Ensure squink has an attribute for historical prices.
        if not hasattr(squink, 'price_history'):
            squink.price_history = []
            
        # Append the current price to the history.
        squink.price_history.append(squink_prc)
        
        # Only compute the EMA if we have enough history (e.g., at least 10 data points)
        if len(squink.price_history) < 10:
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
    
    # @staticmethod
    # def jams(state: Status) -> list[Order]:

    #     current_price = state.maxamt_midprc

    #     orders = []
    #     orders.extend(Strategy.arb(state=state, fair_price=current_price))
    #     orders.extend(Strategy.mm_glft(state=state, fair_price=current_price, mu = -7.60706813499185e-07, sigma = 7.890239872766339e-05, gamma=1e-9, order_amount=10))

    #     return orders
    
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
        #Status.cls_update(state)
        current_price = status.mid

        history = Trade.mid_price_history[CROISSANTS]
        history.append(current_price)
        if len(history) > 8:
            history.pop(0)
            
        if len(history) < 8:
            return []
        
        alpha = 0.1
        ema = history[0]
        for price in history[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        std = np.std(history)
        z_score = (current_price - ema) / std if std > 0 else 0
        
        
        orders = []
        if z_score < -2.5:
            qty = status.possible_buy_amt
            orders.append(Order(CROISSANTS, int(current_price), qty))
        elif z_score > 2.5:
            qty = status.possible_sell_amt
            orders.append(Order(CROISSANTS, int(current_price), -qty))
            
        return orders
    
    @staticmethod
    def voucher_trade(voucher_state: 'Status', strike: float, round_number: int, current_timestamp: int) -> List[Order]:
        return Strategy.voucher_trade(voucher_state, strike, round_number, current_timestamp)
    

    @staticmethod
    def volcanic_rock(state: Status) -> list[Order]:
        current_price = state.mid
        if state.position < 0 and state.position < -int(state.position_limit * 0.5):
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
    state_macarons = Status(MACARON_PRODUCT)

    VOL_TRADE_SIZE = 10
    VOL_POSITION_LIMIT_PER_STRIKE = 100 
    VOL_UNDERLYING_POSITION_LIMIT = 300
    HISTORICAL_MEAN_VOL = 6.86848484848485e-06
    VOL_DIFF_SELL_THRESHOLD = 4.3e-07
    VOL_DIFF_BUY_THRESHOLD  = 1.7e-07
    VOL_DIFF_CLOSE_THRESHOLD = 1e-07

    ## Mac and cheese
    macaron_strategy = MacaronStrategy(logger)


    def __init__(self):
        self.trader_data_cache = {} # Example cache for traderData decoding

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
         # -----------------------------------------------------------------
        # INITIALISE STATIC HELPERS / SIGNAL STATE  (only once)
        # -----------------------------------------------------------------
        if not hasattr(self, "_cam_init"):
            self.cam_sig_pb2: float = 0.0          # Camilla sentiment for Basket‑2
            self.cam_decay        = 0.93           # per‑tick exponential decay
            self.sig_threshold    = 1.0            # threshold to trade
            self._cam_init        = True

        # -----------------------------------------------------------------
        Status.cls_update(state)
        round_number       = 5
        current_timestamp  = state.timestamp
        result: dict[Symbol, list[Order]] = {}
        conversions        = 0

        # Decode traderData
        traderData = {}
        if state.traderData:
            try:
                traderData = jsonpickle.decode(state.traderData)
                # Ensure vol_c_t_history is a list (might be loaded from old format)
                if 'vol_c_t_history' in traderData and not isinstance(traderData['vol_c_t_history'], list):
                    traderData['vol_c_t_history'] = list(traderData['vol_c_t_history'])

            except Exception as e:
                traderData = {}

        # Initialize rolling c_t history as a list if not present
        if 'vol_c_t_history' not in traderData:
            traderData['vol_c_t_history'] = [] # Use a standard list
        vol_hist: list = traderData["vol_c_t_history"]
        self.macaron_strategy.update_state_from_traderdata(traderData)

        result["RAINFOREST_RESIN"] = Trade.resin(self.state_resin)
        result["KELP"] = Trade.kelp(self.state_kelp)
        result["SQUID_INK"] = Trade.ema_mean_reversion(self.state_squink, alpha=0.3, threshold=8)
        result["PICNIC_BASKET1"] = Trade.basket_1(self.state_picnic1, self.state_jam, self.state_djembes, self.state_croiss)
        # result["JAMS"] = Trade.jams(self.state_jam)
        result["PICNIC_BASKET2"] = Trade.basket_2(self.state_picnic2, self.state_jam, self.state_djembes, self.state_croiss)
        pair_orders = Trade.djmb_crs_pair(self.state_djembes, self.state_croiss)
        if "DJEMBES" not in result: result["DJEMBES"] = []
        if "CROISSANTS" not in result: result["CROISSANTS"] = []
        for order in pair_orders:
            if order.symbol == "DJEMBES":
                result["DJEMBES"].append(order)
        #     elif order.symbol == "CROISSANTS":
        #         result["CROISSANTS"].append(order)

        croissant_ema_orders = Trade.croissant_ema(state)
        if "CROISSANTS" not in result: result["CROISSANTS"] = []
        result["CROISSANTS"].extend(croissant_ema_orders)

        ### MACARONS
        # --- Run NEW Macaron Strategy ---
        total_conversions = 0
        try:
             macaron_orders, macaron_conv_req = self.macaron_strategy.run(self.state_macarons, state.timestamp)
             result[MACARON_PRODUCT] = macaron_orders
             total_conversions += macaron_conv_req
        except Exception as e:
             self.macaron_strategy.logger.print(f"ERROR in Macaron Strategy: {e}")
             # Optionally print traceback:
             # import traceback
             # self.macaron_strategy.logger.print(traceback.format_exc())
             result[MACARON_PRODUCT] = [] # Send no orders if error

        # --- New Volcanic Strategy (Round 3 Algorithm - Using List) ---
        volcanic_orders = {}

        # Setup: voucher_symbols, strikes, voucher_states, underlying_state
        voucher_symbols = [
            "VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750",
            "VOLCANIC_ROCK_VOUCHER_10000", "VOLCANIC_ROCK_VOUCHER_10250",
            "VOLCANIC_ROCK_VOUCHER_10500"
        ]
        strikes = {symbol: int(symbol.split('_')[-1]) for symbol in voucher_symbols}
        try:
            voucher_states = {sym: getattr(self, f"state_voucher_{strikes[sym]}") for sym in voucher_symbols}
            underlying_state = self.state_volcanic_rock
        except AttributeError as e:
            logger.print(f"CRITICAL ERROR: Missing volcanic state attribute: {e}")
            final_trader_data = jsonpickle.encode(traderData) # Encode before returning
            logger.flush(state, result, conversions, final_trader_data)
            return result, conversions, final_trader_data

        # 1. Get Inputs (S_t, TTE_t)
        try:
            spot_price = underlying_state.mid
            TTE = compute_time_to_expiry(round_number, current_timestamp)
            logger.print(f"Volcanic Inputs: S_t={spot_price:.2f}, TTE={TTE:.6f} years")

        except Exception as e:
            logger.print(f"Error getting Volcanic inputs or TTE: {e}. Skipping Volcanic strategy this tick.")
            final_trader_data = jsonpickle.encode(traderData)
            logger.flush(state, result, conversions, final_trader_data)
            return result, conversions, final_trader_data

        # 2. Calculate IVs, Moneyness
        moneyness_vol_pairs = []
        sqrt_TTE = math.sqrt(TTE)

        for symbol in voucher_symbols:
            voucher_state = voucher_states[symbol]
            K = strikes[symbol]
            iv = np.nan
            moneyness = np.nan
            try:
                voucher_price = voucher_state.mid
                if voucher_price is None or voucher_price <= 0: continue
                r= 0.0
                iv = implied_volatility(voucher_price, spot_price, K, round_number, current_timestamp, r, tol=7.08e-05, max_iter=1000)

                if not np.isnan(iv) and iv > 1e-9:
                    if K > 0 and spot_price > 0:
                        log_arg = K / spot_price
                        if log_arg > 0:
                            moneyness = math.log(log_arg) / sqrt_TTE
                        else: moneyness = np.nan
                    else: moneyness = np.nan

                    if not np.isnan(moneyness):
                        moneyness_vol_pairs.append((moneyness, iv))

            except Exception as e:
                logger.print(f"Error processing voucher {symbol} in IV/Moneyness loop: {type(e).__name__} - {e!r}")


        # 3. Fit Curve and Get c_t
        
        filtered_pairs = [(m, v) for m, v in moneyness_vol_pairs if not np.isnan(m) and not np.isnan(v)]

        valid_points_count = len(filtered_pairs)
        logger.print(f"Found {valid_points_count} valid & non-NaN pairs for fitting.")

        c_t = np.nan
        coeffs = None

        if valid_points_count >= 3:
            try:
                m_values = np.array([p[0] for p in filtered_pairs])
                sigma_values = np.array([p[1] for p in filtered_pairs])
                coeffs = np.polyfit(m_values, sigma_values, 2)
                c_t = coeffs[2]
                logger.print(f"Quadratic Fit ({valid_points_count} pts): c_t={c_t:.4f}")
            except Exception as e:
                logger.print(f"Error fitting quadratic curve ({valid_points_count} pts): {e}")
                c_t = np.nan
        else:
            logger.print(f"Skipping quadratic fit: Only {valid_points_count} valid points found.")
            c_t = np.nan

        logger.print(f"Final c_t value for tick {state.timestamp}: {c_t}")

        # 4. Compare c_t to Historical Mean (No rolling window or Z-score)
        vol_diff = np.nan
        if not np.isnan(c_t):
            vol_hist.append(float(c_t))
            # keep only the last 200 observations
            if len(vol_hist) > 200:
                del vol_hist[:-200]

        # --------- compute rolling HIST_MEAN --------------------------------
        if len(vol_hist) >= 20:                  # need some history
            historical_mean_vol = float(np.mean(vol_hist))
        else:
            historical_mean_vol = self.HISTORICAL_MEAN_VOL  # fallback to old constant

        # use rolling mean for vol_diff
        vol_diff = np.nan
        if not np.isnan(c_t):
            vol_diff = c_t - historical_mean_vol


        # 5. Choose ATM Strike (C_ATM)
        atm_symbol = None
        min_abs_moneyness = float('inf')

        for symbol in voucher_symbols:
            K = strikes[symbol]
            abs_moneyness = abs(K - spot_price)   # proxy for moneyness when √TTE is constant

            if abs_moneyness < min_abs_moneyness:
                min_abs_moneyness = abs_moneyness
                atm_symbol = symbol

        if atm_symbol:
             logger.print(f"ATM Strike Selected: {atm_symbol} (|m|={min_abs_moneyness:.4f})")
        else:
             logger.print("Could not determine ATM strike.")


        # 6. Trading Signal (based on vol_diff and ATM option)
        if atm_symbol and not np.isnan(vol_diff):
            atm_state = voucher_states[atm_symbol]
            atm_position = atm_state.position

            # Buy Signal (Current Vol Too Low)
            if vol_diff < self.VOL_DIFF_BUY_THRESHOLD:
                if atm_position < self.VOL_POSITION_LIMIT_PER_STRIKE:
                    qty_to_buy = min(self.VOL_TRADE_SIZE, self.VOL_POSITION_LIMIT_PER_STRIKE - atm_position, atm_state.possible_buy_amt)
                    if qty_to_buy > 0:
                        price = atm_state.best_ask
                        if price is not None:
                            order = Order(atm_symbol, int(price), qty_to_buy)
                            if atm_symbol not in volcanic_orders: volcanic_orders[atm_symbol] = []
                            volcanic_orders[atm_symbol].append(order)
                            logger.print(f"TRADE SIGNAL (BUY): {atm_symbol} {qty_to_buy} @ {price} (VolDiff={vol_diff:.4f})")
                        else: logger.print(f"Cannot place BUY for {atm_symbol}: No asks.")
                # else: logger.print(f"BUY Signal for {atm_symbol} blocked: At LONG limit")

            # Sell Signal (Current Vol Too High)
            elif vol_diff > self.VOL_DIFF_SELL_THRESHOLD:
                if atm_position > -self.VOL_POSITION_LIMIT_PER_STRIKE:
                    qty_to_sell = min(self.VOL_TRADE_SIZE, self.VOL_POSITION_LIMIT_PER_STRIKE + atm_position, atm_state.possible_sell_amt)
                    if qty_to_sell > 0:
                        price = atm_state.best_bid
                        if price is not None:
                            order = Order(atm_symbol, int(price), -qty_to_sell)
                            if atm_symbol not in volcanic_orders: volcanic_orders[atm_symbol] = []
                            volcanic_orders[atm_symbol].append(order)
                            logger.print(f"TRADE SIGNAL (SELL): {atm_symbol} {-qty_to_sell} @ {price} (VolDiff={vol_diff:.4f})")
                        else: logger.print(f"Cannot place SELL for {atm_symbol}: No bids.")
                # else: logger.print(f"SELL Signal for {atm_symbol} blocked: At SHORT limit")

            # Close Signal (Current Vol Near Historical Mean)
            elif abs(vol_diff) < self.VOL_DIFF_CLOSE_THRESHOLD:
                if atm_position != 0:
                    qty_to_close = -atm_position
                    price = atm_state.best_bid if atm_position > 0 else atm_state.best_ask
                    possible_trade_qty = atm_state.possible_sell_amt if atm_position > 0 else atm_state.possible_buy_amt
                    if abs(qty_to_close) <= possible_trade_qty:
                         if price is not None:
                             order = Order(atm_symbol, int(price), qty_to_close)
                             if atm_symbol not in volcanic_orders: volcanic_orders[atm_symbol] = []
                             volcanic_orders[atm_symbol].append(order)
                             logger.print(f"TRADE SIGNAL (CLOSE): {atm_symbol} {qty_to_close} @ {price} (|VolDiff|={abs(vol_diff):.4f})")
                         else: logger.print(f"Cannot place CLOSE for {atm_symbol}: No bid/ask.")

        elif not atm_symbol:
            logger.print("No trade signal: ATM symbol not determined.")
        elif np.isnan(vol_diff):
             logger.print("No trade signal: vol_diff is NaN (likely c_t was NaN).")


        # 7. Delta Hedging (Skipped)

        # 8. Risk Constraints Check (Limits checked above)
        result["VOLCANIC_ROCK"] = Trade.ema_mean_reversion(self.state_volcanic_rock, alpha=0.2, threshold=12)
        # --- Merge Volcanic Orders ---
        for symbol, orders in volcanic_orders.items():
             if symbol not in result: result[symbol] = []
             result[symbol].extend(orders)

        logger.print("--- Finished Volcanic Strategy ---")

        # # 4.  CAMILLA‑SIGNAL LOGIC FOR PICNIC_BASKET2
        # -----------------------------------------------------------------
        # (a) update from this tick’s market trades
        for tr in state.market_trades.get("PICNIC_BASKET2", []):
            if tr.buyer == "Charlie":
                self.cam_sig_pb2 =  1.5
            if tr.seller == "Charlie":
                self.cam_sig_pb2 = -1.5

        # (b) decay the signal
        self.cam_sig_pb2 *= self.cam_decay

        # (c) generate incremental orders when |signal| ≥ threshold
        pb2_status = self.state_picnic2
        depth      = state.order_depths["PICNIC_BASKET2"]

        cam_orders: list[Order] = []
        if self.cam_sig_pb2 >= self.sig_threshold and depth.sell_orders:
            price = min(depth.sell_orders)                         # best ask
            qty   = min(1, pb2_status.possible_buy_amt)            # up to 2
            if qty > 0:
                cam_orders.append(Order("PICNIC_BASKET2", price, qty))

        if self.cam_sig_pb2 <= -self.sig_threshold and depth.buy_orders:
            price = max(depth.buy_orders)                          # best bid
            qty   = min(1, pb2_status.possible_sell_amt)           # up to 2
            if qty > 0:
                cam_orders.append(Order("PICNIC_BASKET2", price, -qty))

        #  add Camilla orders to any existing basket‑2 orders
        result.setdefault("PICNIC_BASKET2", []).extend(cam_orders)
        
        
        ### Olivia Squid ink
        if not hasattr(self, "_olivia_init"):
            self.olivia_last_dir: float = 0.0     # +1 buy, –1 sell, 0 neutral
            self._olivia_init = True

        for tr in state.market_trades.get("SQUID_INK", []):
            if tr.buyer == "Olivia":
                self.olivia_last_dir = 1.0
            if tr.seller == "Olivia":
                self.olivia_last_dir = -1.0

        # if Olivia traded this tick, act immediately; else do nothing
        if self.olivia_last_dir != 0.0:
            sq_status = self.state_squink
            depth     = state.order_depths["SQUID_INK"]
            limit     = Status._position_limit["SQUID_INK"]
            current   = state.position.get("SQUID_INK", 0)

            target_pos = limit if self.olivia_last_dir > 0 else -limit
            delta      = target_pos - current

            if delta > 0 and depth.sell_orders:           # need to BUY
                best_ask = min(depth.sell_orders)
                avail    = -depth.sell_orders[best_ask]
                buy_qty  = min(delta, avail)
                if buy_qty > 0:
                    result.setdefault("SQUID_INK", []).append(
                        Order("SQUID_INK", best_ask, buy_qty)
                    )
            elif delta < 0 and depth.buy_orders:          # need to SELL
                best_bid = max(depth.buy_orders)
                avail    = depth.buy_orders[best_bid]
                sell_qty = min(-delta, avail)
                if sell_qty > 0:
                    result.setdefault("SQUID_INK", []).append(
                        Order("SQUID_INK", best_bid, -sell_qty)
                    )

            # reset so we only act once per Olivia print
            self.olivia_last_dir = 0.0
            
        ### Olivia Croissants
        if not hasattr(self, "_olivia_init"):
            self.olivia_last_dir: float = 0.0     # +1 buy, –1 sell, 0 neutral
            self._olivia_init = True

        for tr in state.market_trades.get("CROISSANTS", []):
            if tr.buyer == "Olivia":
                self.olivia_last_dir = 1.0
            if tr.seller == "Olivia":
                self.olivia_last_dir = -1.0

        # if Olivia traded this tick, act immediately; else do nothing
        if self.olivia_last_dir != 0.0:
            sq_status = self.state_squink
            depth     = state.order_depths["CROISSANTS"]
            limit     = Status._position_limit["CROISSANTS"]
            current   = state.position.get("CROISSANTS", 0)

            target_pos = limit if self.olivia_last_dir > 0 else -limit
            delta      = target_pos - current

            if delta > 0 and depth.sell_orders:           # need to BUY
                best_ask = min(depth.sell_orders)
                avail    = -depth.sell_orders[best_ask]
                buy_qty  = min(delta, avail)
                if buy_qty > 0:
                    result.setdefault("CROISSANTS", []).append(
                        Order("CROISSANTS", best_ask, buy_qty)
                    )
            elif delta < 0 and depth.buy_orders:          # need to SELL
                best_bid = max(depth.buy_orders)
                avail    = depth.buy_orders[best_bid]
                sell_qty = min(-delta, avail)
                if sell_qty > 0:
                    result.setdefault("CROISSANTS", []).append(
                        Order("CROISSANTS", best_bid, -sell_qty)
                    )

            # reset so we only act once per Olivia print
            self.olivia_last_dir = 0.0
            

        
        # --- Final Steps ---
        self.macaron_strategy.save_state_to_traderdata(traderData)
        final_trader_data = ""
        try:
            # Encode traderData (without vol_c_t_history)
            final_trader_data = jsonpickle.encode(traderData)
        except Exception as e:
            logger.print(f"Error encoding traderData: {e}")

        logger.flush(state, result, int(round(conversions)), final_trader_data)
        return result, int(round(conversions)), final_trader_data