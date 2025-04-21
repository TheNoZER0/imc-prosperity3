INF = 1e9
EMA_PERIOD = 40
MAX_BASE_QTY = 10


||
    def compute_ema(prices: list[float], period: int) -> float:
        alpha = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema

    @staticmethod
    def squink(state: Status) -> list[Order]:
        required_points = EMA_PERIOD + 1
        hist = state.hist_mid_prc(required_points)
        if len(hist) < required_points:
            return []

        slow_ema = Trade.compute_ema(hist, EMA_PERIOD)
        fast_period = max(2, EMA_PERIOD // 2)
        fast_ema = Trade.compute_ema(hist, fast_period)
        
        macd = fast_ema - slow_ema

        ema_previous = Trade.compute_ema(hist[:-1], EMA_PERIOD)
        ema_current = Trade.compute_ema(hist[1:], EMA_PERIOD)
        derivative = ema_current - ema_previous

        sigma = np.std(hist)
        current = state.mid
        z = (current - slow_ema) / sigma if sigma > 0 else 0

        min_deviation = 1.0
        stop_loss_threshold = 2.5
        max_position_threshold = 20

        orders = []

        if abs(z) >= stop_loss_threshold or abs(state.rt_position) > max_position_threshold:
            minimal_qty = 1
            print("Stop loss condition met: z =", z, "or high position =", state.rt_position)
            print("Trading minimal volume:", minimal_qty)
            if z > 0:
                return [Order(state.product, int(current), -minimal_qty)]
            else:
                return [Order(state.product, int(current), minimal_qty)]
        
        if abs(z) < min_deviation:
            return []

        risk_factor = 1 / (1 + abs(derivative))
        base_qty = max(1, int(MAX_BASE_QTY * risk_factor))
        
        macd_scale = 1.0 + min(0.5, abs(macd))
        base_qty = int(base_qty * macd_scale)
        
        order_scale = min(1.0, abs(z) / 2)
        order_qty = max(1, int(base_qty * order_scale))
        
        print("SQUID_STRAT: z =", z, "derivative =", derivative, "MACD =", macd, 
            "base_qty =", base_qty, "order_qty =", order_qty)
        
        if current < slow_ema:
            orders.append(Order(state.product, int(current), order_qty))
        elif current > slow_ema:
            orders.append(Order(state.product, int(current), -order_qty))
        
        return orders