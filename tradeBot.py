from binance.um_futures import UMFutures
from binance.error import ClientError
from keys import api_key, api_secret # api keys for binance features
import numpy as np

class TradeBot:
    def __init__(self):
        self.client = UMFutures(key=api_key, secret=api_secret, base_url = "https://testnet.binancefuture.com")
        self.symbol = None
        self.__all_symbols = set(map(lambda x: x["symbol"], self.client.book_ticker()))

    def set_symbol(self, symbol: str) -> None:
        assert symbol in self.__all_symbols, "No such symbol"
        self.symbol = symbol
    
    def get_balance(self) -> dict:
        return {coin["asset"]: coin["balance"] for coin in self.client.balance()}
    
    def get_data(self, limit = 50, uni=True) -> np.array:
        data = self.client.klines("BTCUSDT", "1h", limit=limit)
        if uni:
            data = np.array(list(map(lambda x: np.array([float(x[4])]), data)))
        else:
            data = np.array(list(map(lambda x: np.array([float(x[4]), float(x[2])-float(x[3]), float(x[8]), float(x[5])]), data)))
        return data

    def new_order(self, quantity: float, side="BUY", type="MARKET") -> bool:
        assert quantity > 0, "Quantity must be positive number"
        assert self.symbol is not None, "Symbol must be a string"
        try:
            response = self.client.new_order(
                symbol=self.symbol,
                side=side,
                type=type,
                quantity=quantity,
            )
            print(f"Bot {side} quantity: {quantity}")
            return True
        except ClientError as error:
            print(
                "Found error. status: {}, error code: {}, error message: {}".format(
                    error.status_code, error.error_code, error.error_message
                )
            )
            return False
    
    def current_price(self) -> float:
        return float(self.client.mark_price(self.symbol)["markPrice"])
