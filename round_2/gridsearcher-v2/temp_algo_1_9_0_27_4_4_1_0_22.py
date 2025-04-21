import json
import math
import numpy as np
import json
from typing import Any, List, Dict
import jsonpickle
from datamodel import *

# =============================================================================
#                                        ROUND 2
# =============================================================================
# ----- Parameters and Constants -----
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

# MIN_ARB_DIFF = 9
# SCALING_FACTOR = 0.27
# MAX_BASE_QTY = 4
# EMA_PERIOD = 4
# Z_THRESHOLD = 1.0
# VOL_WINDOW = 22

MIN_ARB_DIFF = 8
SCALING_FACTOR = 0.5
MAX_BASE_QTY = 10

VOL_WINDOW = 20

EMA_PERIOD = 10
Z_THRESHOLD = 1.5
BASE_TICK = 1
VOL_SCALE = 0.1

mid_price_history = {sym: [] for sym in [CROISSANTS, JAMS, DJEMBES, PICNIC_BASKET1, PICNIC_BASKET2]}

class Strategy_R2:

    def __init__(self, state: TradingState):
        self.state = state
        self.mid_price_history = {
            CROISSANTS: [],
            JAMS: [],
            DJEMBES: [],
            PICNIC_BASKET1: [],
            PICNIC_BASKET2: []
        }

    def get_mid_price(state: TradingState, symbol: str) -> float:
        od = state.order_depths[symbol]
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else 0
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else (best_bid + 1)
        return (best_bid + best_ask) / 2

    def update_mid_price_history(state: TradingState, symbol: str) -> list[float]:
        mid = Strategy_R2.get_mid_price(state, symbol)
        hist = mid_price_history[symbol]
        hist.append(mid)
        if len(hist) > VOL_WINDOW:
            hist.pop(0)
        return hist


    def estimate_volatility(prices: list[float]) -> float:
        return np.std(prices) if len(prices) >= 2 else 0
    
    def theoretical_basket1_price(croissant_price: float, jam_price: float, djembe_price: float) -> float:
        return 6 * croissant_price + 3 * jam_price + djembe_price


    def theoretical_basket2_price(croissant_price: float, jam_price: float) -> float:
        return 4 * croissant_price + 2 * jam_price


    def arbitrage_strategy(state: TradingState, base_qty: int = MAX_BASE_QTY) -> list[Order]:
        orders = []
        croissant_mid = Strategy_R2.get_mid_price(state, CROISSANTS)
        jam_mid = Strategy_R2.get_mid_price(state, JAMS)
        djembe_mid = Strategy_R2.get_mid_price(state, DJEMBES)

        basket1_theoretical = Strategy_R2.theoretical_basket1_price(croissant_mid, jam_mid, djembe_mid)
        basket2_theoretical = Strategy_R2.theoretical_basket2_price(croissant_mid, jam_mid)

        basket1_mid = Strategy_R2.get_mid_price(state, PICNIC_BASKET1)
        basket2_mid = Strategy_R2.get_mid_price(state, PICNIC_BASKET2)

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
        fair_price = Strategy_R2.get_mid_price(state, symbol)
        hist = Strategy_R2.update_mid_price_history(state, symbol)
        vol = Strategy_R2.estimate_volatility(hist)
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


    def compute_ema(prices: list[float], period: int) -> float:
        if not prices:
            return 0
        alpha = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema


    def ema_strategy(state: TradingState, symbol: str, period: int = EMA_PERIOD, z_threshold: float = Z_THRESHOLD) -> list[Order]:
        """Generate orders based on an EMA mean-reversion signal when the market is quiet."""
        hist = Strategy_R2.update_mid_price_history(state, symbol)
        if len(hist) < period:
            return []
        ema_val = Strategy_R2.compute_ema(hist, period)
        std = np.std(hist) if len(hist) > 1 else 0
        current = hist[-1]
        z = (current - ema_val) / std if std != 0 else 0

        orders = []
        # If |z| is below threshold, issue a minimal order to capture mean reversion.
        if abs(z) < z_threshold:
            # If current price is below EMA, suggest a small buy; if above, a small sell.
            if current < ema_val:
                orders.append(Order(symbol, int(current), 1))
            elif current > ema_val:
                orders.append(Order(symbol, int(current), -1))
        return orders



# =============================================================================
#                         ROUND 1: Visual/Status Strategy (R1)
# =============================================================================

# FYI I renamed Round 1 classes with an _R1 suffix for clarity and to avoid name collisions.
INF = 1e9

class Status_R1:
    _position_limit = {
        "RAINFOREST_RESIN":50,
        "KELP":50,
        "SQUID_INK":50
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
        cls._state = state
        # Only update products that belong to Round 1 assets.
        for product, posit in state.position.items():
            if product in cls._position_limit:
                cls._realtime_position[product] = posit
        for product, orderdepth in state.order_depths.items():
            if product not in cls._position_limit:
                continue
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

class Strategy_R1:
    @staticmethod
    def arb(state: Status_R1, fair_price):
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
        state: Status_R1,
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
        state: Status_R1,
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
    def convert(state: Status_R1):
        if state.position < 0:
            return -state.position
        elif state.position > 0:
            return -state.position
        else:
            return 0


class Trade_R1:
    @staticmethod   
    def resin(state: Status_R1) -> list[Order]:

        current_price = state.maxamt_midprc

        orders = []
        orders.extend(Strategy_R1.arb(state=state, fair_price=current_price))
        orders.extend(Strategy_R1.mm_ou(state=state, fair_price=current_price, gamma=1e-9, order_amount=50))

        return orders
    
    @staticmethod
    def kelp(state: Status_R1) -> list[Order]:

        current_price = state.maxamt_midprc

        orders = []
        orders.extend(Strategy_R1.arb(state=state, fair_price=current_price))
        orders.extend(Strategy_R1.mm_glft(state=state, fair_price=current_price, mu = 1.2484084052394708e-07, sigma = 0.0001199636554242691, gamma=1e-9, order_amount=50))

        return orders
    
    @staticmethod
    def squink(state: Status_R1) -> list[Order]:

        current_price = state.maxamt_midprc

        orders = []
        orders.extend(Strategy_R1.arb(state=state, fair_price=current_price))

        return orders
    
    @staticmethod
    def convert(state: Status_R1) -> int:
        return Strategy_R1.convert(state=state)

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

# =============================================================================
#                         Chunky Trader Class
# =============================================================================

class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0
        trader_data = "Round2 + Round1"
        
        # ----- Run Round 2 Strategy -----
        result_round2 = {symbol: [] for symbol in [CROISSANTS, JAMS, DJEMBES, PICNIC_BASKET1, PICNIC_BASKET2]}
        # Update mid-price history for PICNIC_BASKET1.
        hist = Strategy_R2.update_mid_price_history(state, PICNIC_BASKET1)
        use_ema_strategy = False
        if len(hist) >= EMA_PERIOD:
            ema_val = Strategy_R2.compute_ema(hist, period=EMA_PERIOD)
            std = np.std(hist) if len(hist) > 1 else 0
            current = hist[-1]
            z = (current - ema_val) / std if std != 0 else 0

            # Compute the derivative of the normalised z-score over the last two intervals.
            if len(hist) >= EMA_PERIOD + 1:
                # Normalise the last two differences.
                prev_z = (hist[-2] - Strategy_R2.compute_ema(hist[:-1], period=EMA_PERIOD)) / (np.std(hist[:-1]) + 1e-6)
                derivative = z - prev_z
            else:
                derivative = 0

            # Log diagnostic information 🤓
            logger.print("Round2 Diagnostics:",
                         "EMA =", ema_val,
                         "std =", std,
                         "current =", current,
                         "z =", z,
                         "derivative =", derivative)

            # Use EMA mode if the market has a small normalised z
            # ORRR if the derivative has just reversed (derivative has opposite sign compared to previous derivative)
            if abs(z) < Z_THRESHOLD or derivative * (prev_z if 'prev_z' in locals() else 1) < 0:
                use_ema_strategy = True

        if use_ema_strategy:
            orders_r2 = Strategy_R2.ema_strategy(state, PICNIC_BASKET1, period=EMA_PERIOD, z_threshold=Z_THRESHOLD)
            for o in orders_r2:
                result_round2.setdefault(o.symbol, []).append(o)
                logger.print("Round2 EMA order:", o.symbol, o.price, o.quantity)
        else:
            orders_r2 = Strategy_R2.arbitrage_strategy(state, base_qty=MAX_BASE_QTY)
            for o in orders_r2:
                result_round2.setdefault(o.symbol, []).append(o)
                logger.print("Round2 Arb order:", o.symbol, o.price, o.quantity)
            for basket in [PICNIC_BASKET1, PICNIC_BASKET2]:
                orders_mm = Strategy_R2.advanced_market_making_strategy(state, basket, base_order_size=10)
                for o in orders_mm:
                    result_round2.setdefault(o.symbol, []).append(o)
                    logger.print("Round2 MM order for", basket, ":", o.symbol, o.price, o.quantity)
        result.update(result_round2)


        # ----- Run Round 1 Strategy -----
        # For Round 1 assets: RAINFOREST_RESIN, KELP, and SQUID_INK.
        result_round1 = {}
        for asset in ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]:
            status = Status_R1(asset)
            Status_R1.cls_update(state)
            if asset == "RAINFOREST_RESIN":
                orders_r1 = Trade_R1.resin(status)
            elif asset == "KELP":
                orders_r1 = Trade_R1.kelp(status)
            elif asset == "SQUID_INK":
                orders_r1 = Trade_R1.squink(status)
            else:
                orders_r1 = []
            result_round1[asset] = orders_r1
            for o in orders_r1:
                logger.print("Round1 order for", asset, ":", o.symbol, o.price, o.quantity)
        result.update(result_round1)

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data