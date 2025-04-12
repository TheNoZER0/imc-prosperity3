import json
from typing import Dict, List
from json import JSONEncoder
import jsonpickle

Time = int
Symbol = str
Product = str
Position = int
UserId = str
ObservationValue = int


class Listing:
    """
    Represents a market listing for a product.

    Properties:
    - symbol (Symbol): The unique identifier for the product.
    - product (Product): The name or type of the product.
    - denomination (Product): The denomination associated with the product.
    """

    def __init__(self, symbol: Symbol, product: Product, denomination: Product):
        self.symbol = symbol
        self.product = product
        self.denomination = denomination


class ConversionObservation:
    """
    Represents the details for a product's conversion observation.

    Properties:
    - bidPrice (float): The price at which the product is being bought.
    - askPrice (float): The price at which the product is being sold.
    - transportFees (float): The transportation fees for the product.
    - exportTariff (float): The export tariff for the product.
    - importTariff (float): The import tariff for the product.
    - sugarPrice (float): The price of sugar, relevant to the product.
    - sunlightIndex (float): The sunlight index affecting the product.
    """

    def __init__(self, bidPrice: float, askPrice: float, transportFees: float, exportTariff: float, importTariff: float,
                 sugarPrice: float, sunlightIndex: float):
        self.bidPrice = bidPrice
        self.askPrice = askPrice
        self.transportFees = transportFees
        self.exportTariff = exportTariff
        self.importTariff = importTariff
        self.sugarPrice = sugarPrice
        self.sunlightIndex = sunlightIndex


class Observation:
    """
    Represents the observation data for products and their conversions.

    Properties:
    - plainValueObservations (Dict[Product, ObservationValue]): A dictionary of product observations (simple values).
    - conversionObservations (Dict[Product, ConversionObservation]): A dictionary of product conversion observations.
    """

    def __init__(self, plainValueObservations: Dict[Product, ObservationValue],
                 conversionObservations: Dict[Product, ConversionObservation]) -> None:
        self.plainValueObservations = plainValueObservations
        self.conversionObservations = conversionObservations

    def __str__(self) -> str:
        return f"(plainValueObservations: {jsonpickle.encode(self.plainValueObservations)}, " \
               f"conversionObservations: {jsonpickle.encode(self.conversionObservations)})"


class Order:
    """
    Represents an order in the market.

    Properties:
    - symbol (Symbol): The product symbol associated with the order.
    - price (int): The price at which the algorithm wants to buy or sell.
    - quantity (int): The quantity of the product to buy or sell (positive for buy, negative for sell).
    """

    def __init__(self, symbol: Symbol, price: int, quantity: int) -> None:
        self.symbol = symbol
        self.price = price
        self.quantity = quantity

    def __str__(self) -> str:
        return f"({self.symbol}, {self.price}, {self.quantity})"

    def __repr__(self) -> str:
        return f"({self.symbol}, {self.price}, {self.quantity})"


class OrderDepth:
    """
    Represents the order depth for a specific symbol in the market.

    Properties:
    - buy_orders (Dict[int, int]): A dictionary of buy orders, where the key is the price and the value is the quantity.
    - sell_orders (Dict[int, int]): A dictionary of sell orders, where the key is the price and the value is the quantity (negative).
    """

    def __init__(self):
        self.buy_orders: Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}


class Trade:
    """
    Represents a trade in the market.

    Properties:
    - symbol (Symbol): The product symbol involved in the trade.
    - price (int): The price at which the trade was executed.
    - quantity (int): The quantity of the product in the trade.
    - buyer (UserId, optional): The identifier of the buyer in the trade.
    - seller (UserId, optional): The identifier of the seller in the trade.
    - timestamp (Time): The timestamp of the trade.
    """
    def __init__(self, symbol: Symbol, price: int, quantity: int, buyer: UserId = None, seller: UserId = None,
                 timestamp: int = 0) -> None:
        self.symbol = symbol
        self.price = price
        self.quantity = quantity
        self.buyer = buyer
        self.seller = seller
        self.timestamp = timestamp

    def __str__(self) -> str:
        return f"({self.symbol}, {self.buyer} << {self.seller}, {self.price}, {self.quantity}, {self.timestamp})"

    def __repr__(self) -> str:
        return f"({self.symbol}, {self.buyer} << {self.seller}, {self.price}, {self.quantity}, {self.timestamp})"


class TradingState:
    """
    Represents the current state of trading, including all relevant data.

    Properties:
    - traderData (str): Data specific to the trader.
    - timestamp (Time): The current timestamp of the trading state.
    - listings (Dict[Symbol, Listing]): The listings for each product symbol.
    - order_depths (Dict[Symbol, OrderDepth]): The order depths for each product symbol.
    - own_trades (Dict[Symbol, List[Trade]]): The trades executed by the trader.
    - market_trades (Dict[Symbol, List[Trade]]): The trades in the market.
    - position (Dict[Product, Position]): The trader's positions in various products.
    - observations (Observation): The latest observations regarding the market and conversions.
    """
    def __init__(self,
                 traderData: str,
                 timestamp: Time,
                 listings: Dict[Symbol, Listing],
                 order_depths: Dict[Symbol, OrderDepth],
                 own_trades: Dict[Symbol, List[Trade]],
                 market_trades: Dict[Symbol, List[Trade]],
                 position: Dict[Product, Position],
                 observations: Observation):
        self.traderData = traderData
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations

    def toJSON(self):
        """
        Converts the trading state to a JSON string.

        Returns:
        - str: JSON representation of the trading state.
        """
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)


class ProsperityEncoder(JSONEncoder):
    """
    Custom JSON encoder for Prosperity objects.

    Converts complex objects to their dictionary representations for JSON serialization.
    """

    def default(self, o):
        return o.__dict__
