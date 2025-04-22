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
        logger.print(f"Fitted Parabola Coeffs: {coeffs}, Base IV = {base_IV:.2f}")
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
    logger.print(f"[Delta Info] For strike {strike}: Delta = {delta:.2f}")
    # *** END OF DELTA COMPUTATION SNIPPET ***

    # Now, use a dynamic buffer based on volatility to set target buy/sell thresholds:
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
EMA_PERIOD = 13
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
        #Status.cls_update(state)
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
    
    @staticmethod
    
    def macaron_volume(state: Status, vol_thresh: float = 1.9) -> List[Order]:
        orders = []
        bid_vol = sum(amt for _, amt in state.bids)
        ask_vol = sum(-amt for _, amt in state.asks)

        # If bids out‑size asks by 20%+, go long at the ask
        if bid_vol > vol_thresh * ask_vol and state.possible_buy_amt > 0:
            orders.append(Order(
                state.product,
                state.best_ask,
                state.possible_buy_amt
            ))

        # If asks out‑size bids by 20%+, go short at the bid
        elif ask_vol > vol_thresh * bid_vol and state.possible_sell_amt > 0:
            orders.append(Order(
                state.product,
                state.best_bid,
                -state.possible_sell_amt
            ))

        return orders

    
    @staticmethod
    def macarons_parity(state: Status, current_conversions: int) -> tuple[list[Order], int]:
        """
        Local ↔ foreign parity arb for MAGNIFICENT_MACARONS,
        widened by PARITY_MARGIN to cover slippage & fees.
        """
        STORAGE_COST = 0.1
        obs = state._state.observations.conversionObservations["MAGNIFICENT_MACARONS"]

        implied_bid = (
            obs.bidPrice
            - obs.exportTariff
            - obs.transportFees
            - STORAGE_COST
        )
        implied_ask = (
            obs.askPrice
            + obs.importTariff
            + obs.transportFees
            + STORAGE_COST
        )

        orders: list[Order] = []
        used_conv = 0
        remaining_conv = max(0, 10 - current_conversions)

        # buy local if below implied foreign‐bid minus margin
        for ask_price, ask_amt in state.asks:
            if (
                ask_price < implied_bid - PARITY_MARGIN
                and remaining_conv > 0
            ):
                qty = min(-ask_amt, state.possible_buy_amt, remaining_conv)
                orders.append(Order("MAGNIFICENT_MACARONS", int(ask_price), qty))
                used_conv += qty
                remaining_conv -= qty

        # sell local if above implied foreign‐ask plus margin
        for bid_price, bid_amt in state.bids:
            if (
                bid_price > implied_ask + PARITY_MARGIN
                and remaining_conv > 0
            ):
                qty = min(bid_amt, state.possible_sell_amt, remaining_conv)
                orders.append(Order("MAGNIFICENT_MACARONS", int(bid_price), -qty))
                used_conv += qty
                remaining_conv -= qty

        # unwind adverse positions if they run away
        pos = state.position
        mid = state.mid
        STOP_THRESH = max(1, int((implied_ask - implied_bid) * 2))

        if pos > 0 and mid > implied_ask + STOP_THRESH:
            orders.append(Order("MAGNIFICENT_MACARONS", state.best_bid, -pos))
        elif pos < 0 and mid < implied_bid - STOP_THRESH:
            orders.append(Order("MAGNIFICENT_MACARONS", state.best_ask, -pos))

        return orders, used_conv
        
    @staticmethod
    def convert_macarons(state: Status) -> int:
        """
        Enforce conversion limit up to 10 units per tick to reset position.
        """
        pos = state.position
        if pos > 0:
            return min(pos, 10)
        elif pos < 0:
            return max(pos, -10)
        return 0

    
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
    state_macarons = Status("MAGNIFICENT_MACARONS")

    last_vol_coeffs = None
    voucher_deltas = {}
    
    VOL_PARAMS = {
        "std_window": 45,
        #"mean_volatility": {
           # '9500': 0.129, '9750': 0.159, '10000': 0.149, '10250': 0.138, '10500': 0.142 # Use STRINGS
        #},
        "zscore_threshold": 1, # default 2
        "trade_size": 200, # default 20
        "min_iv_history": 10
    }

    def __init__(self):
        self.offsets = list(range(-5, 2))   # try –5 up to +1
        self.counts  = {o:1 for o in self.offsets}
        self.rewards = {o:0.0 for o in self.offsets}

    def choose_offset(self):
        total = sum(self.counts.values())
        import math
        def ucb(o):
            return (self.rewards[o]/self.counts[o]
                    + math.sqrt(2*math.log(total)/self.counts[o]))
        return max(self.offsets, key=ucb)

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        Status.cls_update(state)
        round_number = 5
        current_timestamp = state.timestamp

        result = {}

        #Round 1 orders:
        result["RAINFOREST_RESIN"] = Trade.resin(self.state_resin)
        result["KELP"] = Trade.kelp(self.state_kelp)
        result["SQUID_INK"] = Trade.ema_mean_reversion(self.state_squink)

        # Round 2 orders:
        result["PICNIC_BASKET1"] = Trade.basket_1(self.state_picnic1, self.state_jam, self.state_djembes, self.state_croiss)
        #result["JAMS"] = Trade.jams(self.state_jam)
        result["PICNIC_BASKET2"] = Trade.basket_2(self.state_picnic2, self.state_jam, self.state_djembes, self.state_croiss)
        #result["DJEMBES"] = Trade.djmb_crs_pair(self.state_djembes, self.state_croiss)
        result["CROISSANTS"] = Trade.croissant_ema(self.state_croiss)
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
        
        # --- Decode TraderData ---
        traderData = {}
        if state.traderData:
            try:
                traderData = jsonpickle.decode(state.traderData)
            except Exception as e:
                logger.print(f"Error decoding traderData: {e}")
                traderData = {} # Start fresh if decode fails
        # --- End Decode ---

        
        
        strikes = {
            "VOLCANIC_ROCK_VOUCHER_9500": 9500, "VOLCANIC_ROCK_VOUCHER_9750": 9750,
            "VOLCANIC_ROCK_VOUCHER_10000": 10000, "VOLCANIC_ROCK_VOUCHER_10250": 10250,
            "VOLCANIC_ROCK_VOUCHER_10500": 10500
        }
        
        voucher_symbols = [
            "VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750",
            "VOLCANIC_ROCK_VOUCHER_10000", "VOLCANIC_ROCK_VOUCHER_10250",
            "VOLCANIC_ROCK_VOUCHER_10500"
        ]
        
        strikes = { voucher_symbols[i]: int(voucher_symbols[i].split('_')[-1]) for i in range(len(voucher_symbols))}

        # --- Initialize traderData structures if first run ---
        if 'vol_iv_history' not in traderData:
            traderData['vol_iv_history'] = {str(k): [] for k in strikes.values()}
        # --- End Initialize ---
        
        try:
            # Dynamically get voucher states using getattr
            voucher_states = {sym: getattr(self, f"state_voucher_{strikes[sym]}") for sym in voucher_symbols}
            # Ensure the underlying state exists
            if not hasattr(self, 'state_volcanic_rock'):
                 raise AttributeError("state_volcanic_rock not defined in Trader")
            underlying_state = self.state_volcanic_rock
        except AttributeError as e:
             logger.print(f"CRITICAL ERROR: Missing state attribute in Trader class: {e}")
             logger.flush(state, result, 0, jsonpickle.encode(traderData))
             return result, 0, jsonpickle.encode(traderData) # Stop processing
        
        # 1. Get Common Inputs for Volatility Calculations
        try:
            spot_volcanic = underlying_state.mid # Underlying price (St)
            if spot_volcanic is None or spot_volcanic <= 0:
                 raise ValueError(f"Invalid spot price for VOLCANIC_ROCK: {spot_volcanic}")
            TTE = compute_time_to_expiry(round_number, current_timestamp) # Time to Expiry
            r = 0.0 # Risk-free rate
            if TTE <= 1e-9: raise ValueError(f"Time to expiry too small: {TTE}")
        except Exception as e:
            logger.print(f"Error getting Volcanic inputs or TTE: {e}. Skipping Volcanic strategy.")
            # Persist data before returning
            traderData['voucher_deltas'] = self.voucher_deltas
            final_trader_data = jsonpickle.encode(traderData)
            logger.flush(state, result, 0, final_trader_data)
            return result, 0, final_trader_data

        net_voucher_delta_change = 0.0 # Tracks delta change from *new* voucher trades this tick
        temp_voucher_orders: Dict[Symbol, List[Order]] = {sym: [] for sym in voucher_symbols} # Store voucher orders before adding to result
        current_ivs = {} # Store current IVs {strike_key: iv} for potential later use

        # 2. Calculate IV, Z-Score, Delta and Generate Voucher Orders (per strike)
        logger.print("--- Volatility Calculation Start ---")
        for symbol in voucher_symbols:
            voucher_state = voucher_states[symbol]
            K = strikes[symbol]
            strike_key = str(K) # Use string representation of strike for dictionary keys

            try:
                market_price = voucher_state.mid # Current mid-price of the voucher

                if market_price is None or market_price <= 0:
                    logger.print(f"Skipping {symbol}: Invalid market price ({market_price})")
                    current_ivs[strike_key] = np.nan
                    self.voucher_deltas.pop(symbol, None) # Remove potentially stale delta
                    continue

                # Calculate Intrinsic Value
                intrinsic_value = max(spot_volcanic - K * math.exp(-r * TTE), 0.0)
                tolerance = 0.01 # Small tolerance for near-intrinsic checks

                actual_v_t = np.nan # Default IV to NaN

                # Check if price is too close to or below intrinsic value
                if market_price <= (intrinsic_value + tolerance):
                    logger.print(f"{symbol}: Market price {market_price:.2f} near/below intrinsic {intrinsic_value:.2f}. Treating IV as near zero.")
                    # Option 1: Assign NaN (preferred, avoids potentially misleading Z-score)
                    actual_v_t = np.nan
                    # Option 2: Assign a tiny volatility (use if delta calc needs a non-NaN value)
                    # actual_v_t = 1e-5
                else:
                    # Calculate Implied Volatility only if price is safely above intrinsic
                    actual_v_t = implied_volatility(market_price, spot_volcanic, K, round_number, current_timestamp, r)
                    logger.print(f"IV Calc {symbol}: S={spot_volcanic:.2f}, K={K}, T={TTE:.4f}, V={market_price:.2f} -> IV={actual_v_t}")

                current_ivs[strike_key] = actual_v_t # Store calculated IV

                # Update IV History and Calculate Z-Score only if IV is valid
                if not np.isnan(actual_v_t) and actual_v_t > 1e-6: # Check for valid, non-zero IV
                    # --- Update IV History ---
                    iv_history = traderData['vol_iv_history'].get(strike_key, [])
                    iv_history.append(actual_v_t)
                    # Limit history length to the rolling window size
                    max_hist_len = self.VOL_PARAMS["std_window"]
                    if len(iv_history) > max_hist_len:
                        iv_history = iv_history[-max_hist_len:]
                    traderData['vol_iv_history'][strike_key] = iv_history
                    # --- End Update IV History ---

                    # --- Calculate Z-Score and Generate Orders ---
                    min_hist_len = self.VOL_PARAMS["min_iv_history"]
                    if len(iv_history) >= min_hist_len: # Ensure enough data points
                        std_dev = np.std(iv_history)
                        # *** Calculate Rolling Mean Implied Volatility ***
                        rolling_mean_vol = np.mean(iv_history)
                        # *************************************************

                        logger.print(f"{symbol}: IV={actual_v_t:.4f}, RollingMean={rolling_mean_vol:.4f}, StdDev={std_dev:.4f}, HistLen={len(iv_history)}")

                        if std_dev > 1e-6: # Avoid division by zero/tiny std dev
                            vol_z_score = (actual_v_t - rolling_mean_vol) / std_dev
                            logger.print(f"{symbol}: Z-Score = ({actual_v_t:.4f} - {rolling_mean_vol:.4f}) / {std_dev:.4f} = {vol_z_score:.2f}")

                            # --- Generate Orders based on Z-Score ---
                            trade_qty = self.VOL_PARAMS["trade_size"]
                            z_thresh = self.VOL_PARAMS["zscore_threshold"]

                            if vol_z_score > z_thresh: # IV significantly high -> Sell Voucher
                                qty_to_sell = min(trade_qty, voucher_state.possible_sell_amt)
                                if qty_to_sell > 0:
                                    order_price = voucher_state.best_bid # Hit best bid
                                    if order_price is not None:
                                        order = Order(symbol, order_price, -qty_to_sell)
                                        temp_voucher_orders[symbol].append(order)
                                        logger.print(f"-> VOL SELL {symbol} @ {order_price} x {qty_to_sell} (Z={vol_z_score:.2f})")
                                        # Accumulate delta change for hedging
                                        delta_for_hedge = calculate_delta(spot_volcanic, K, TTE, r, actual_v_t) # Use current IV for delta
                                        net_voucher_delta_change += delta_for_hedge * order.quantity
                                    else: logger.print(f"Cannot SELL {symbol}: No bids available")

                            elif vol_z_score < -z_thresh: # IV significantly low -> Buy Voucher
                                qty_to_buy = min(trade_qty, voucher_state.possible_buy_amt)
                                if qty_to_buy > 0:
                                    order_price = voucher_state.best_ask # Hit best ask
                                    if order_price is not None:
                                        order = Order(symbol, order_price, qty_to_buy)
                                        temp_voucher_orders[symbol].append(order)
                                        logger.print(f"-> VOL BUY {symbol} @ {order_price} x {qty_to_buy} (Z={vol_z_score:.2f})")
                                        # Accumulate delta change for hedging
                                        delta_for_hedge = calculate_delta(spot_volcanic, K, TTE, r, actual_v_t) # Use current IV for delta
                                        net_voucher_delta_change += delta_for_hedge * order.quantity
                                    else: logger.print(f"Cannot BUY {symbol}: No asks available")
                            # --- End Generate Orders ---
                        else:
                            logger.print(f"{symbol}: Std Dev ({std_dev:.4f}) too small, skipping Z-score signal.")
                    else:
                         logger.print(f"{symbol}: Not enough IV history ({len(iv_history)} < {min_hist_len}) to calculate Z-score.")
                    # --- End Calculate Z-Score ---

                    # --- Calculate and Store Delta (always update if IV is valid) ---
                    delta = calculate_delta(spot_volcanic, K, TTE, r, actual_v_t)
                    self.voucher_deltas[symbol] = delta # Store the calculated delta for this symbol
                    logger.print(f"[Delta Info] {symbol} (Strike {K}): Stored Delta = {delta:.4f}")
                    # --- End Delta Calc ---

                else: # Handle cases where IV was NaN or near zero
                    logger.print(f"{symbol}: IV calculation failed, NaN, or near zero. Skipping Z-score and setting delta based on moneyness.")
                    # Set delta based purely on moneyness if IV is invalid
                    delta = 1.0 if spot_volcanic > K else 0.0
                    self.voucher_deltas[symbol] = delta # Store approximate delta
                    logger.print(f"[Delta Info] {symbol} (Strike {K}): Stored Approx. Delta = {delta:.4f}")


            except Exception as e:
                logger.print(f"ERROR processing {symbol} (strike {K}): {type(e).__name__} - {e!r}")
                self.voucher_deltas.pop(symbol, None) # Remove delta if error occurred during processing
                # Optionally add traceback:
                # import traceback
                # logger.print(traceback.format_exc())

        logger.print("--- Volatility Calculation End ---")
        # --- End Voucher Processing Loop ---


        # 3. Calculate Net Portfolio Delta and Hedge Order
        # current_portfolio_delta = 0.0
        # # Sum delta of existing positions
        # for symbol in voucher_symbols:
        #     # Ensure position exists before accessing
        #     position = voucher_states[symbol].position if symbol in state.position else 0
        #     delta = self.voucher_deltas.get(symbol, 0.0) # Use stored delta
        #     current_portfolio_delta += position * delta

        # # Total delta after planned trades
        # total_target_delta = current_portfolio_delta + net_voucher_delta_change

        # # Calculate desired hedge position & order quantity
        # target_hedge_position = -round(total_target_delta)
        # current_hedge_position = self.state_volcanic_rock.position
        # hedge_order_qty = target_hedge_position - current_hedge_position

        # # Apply VOLCANIC_ROCK position limits
        # hedge_limit = self.state_volcanic_rock.position_limit
        # if hedge_order_qty > 0: # Buying hedge
        #     hedge_order_qty = min(hedge_order_qty, hedge_limit - current_hedge_position)
        # elif hedge_order_qty < 0: # Selling hedge
        #     hedge_order_qty = max(hedge_order_qty, -hedge_limit - current_hedge_position)

        # logger.print(f"Delta Hedge: CurrentVoucherDelta={current_portfolio_delta:.2f}, NewTradeDelta={net_voucher_delta_change:.2f}, TargetNetDelta={total_target_delta:.2f}")
        # logger.print(f"Hedge Calc: TargetHedgePos={target_hedge_position}, CurrentHedgePos={current_hedge_position}, OrderQty={hedge_order_qty}")
        result["VOLCANIC_ROCK"] = Trade.ema_mean_reversion(self.state_volcanic_rock, alpha=0.2, threshold=12)
        # Create hedge order for VOLCANIC_ROCK if needed
        # if abs(hedge_order_qty) > 0:
        #     best_bid_vr = self.state_volcanic_rock.best_bid
        #     best_ask_vr = self.state_volcanic_rock.best_ask
        #     hedge_price = None

        #     if hedge_order_qty > 0 and best_ask_vr is not None: # Buying hedge
        #          hedge_price = best_ask_vr # Hit best ask
        #     elif hedge_order_qty < 0 and best_bid_vr is not None: # Selling hedge
        #          hedge_price = best_bid_vr # Hit best bid

        #     if hedge_price is not None : # Place order only if price is valid
        #         hedge_order = Order("VOLCANIC_ROCK", hedge_price, hedge_order_qty)
        #         logger.print(f"-> HEDGE ORDER: {hedge_order.symbol} @ {hedge_order.price} x {hedge_order.quantity}")
        #         if "VOLCANIC_ROCK" not in result: result["VOLCANIC_ROCK"] = []
        #         result["VOLCANIC_ROCK"].append(hedge_order)
        #     else:
        #         logger.print(f"Cannot place hedge order: Invalid market BBO for VOLCANIC_ROCK")
        # else:
        #      logger.print("No hedge order needed.")


        # --- Combine Orders ---
        # Add generated voucher orders to the main result dictionary
        for symbol, orders in temp_voucher_orders.items():
            if orders:
                 if symbol not in result: result[symbol] = []
                 result[symbol].extend(orders)
        # --- End Combine ---

        foreign_ask = state.observations.conversionObservations["MAGNIFICENT_MACARONS"].askPrice
        offset = self.choose_offset()
        px     = int(foreign_ask + offset)
        qty    = min(100, self.state_macarons.possible_sell_amt)

        # place local sell:
        order = Order("MAGNIFICENT_MACARONS", px, -qty)
        result["MAGNIFICENT_MACARONS"] = [order]

        # … after the exchange, measure how many actually filled …
        conv = state.observations.conversionObservations["MAGNIFICENT_MACARONS"]
        cost_per_unit = conv.askPrice + conv.transportFees + conv.importTariff
        filled = min(qty, self.state_macarons.best_bid_amount)  # approximate
        profit = (px - cost_per_unit) * filled

        # update bandit stats:
        self.counts[offset]  += 1
        self.rewards[offset] += profit
        # --- Final Steps ---
        conversions = 0 # Adjust if needed
        final_trader_data = ""
        try:
            final_trader_data = jsonpickle.encode(traderData)
        except Exception as e:
            logger.print(f"Error encoding traderData: {e}")

        logger.flush(state, result, conversions, final_trader_data)
        return result, conversions, final_trader_data