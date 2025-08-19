from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest ,StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd


# AlpacaTrading class provides methods to interact with Alpaca's trading and historical data APIs
class AlpacaTrading:
    def __init__(self, api_key, secret_key):
        # Initialize trading and historical data clients using API keys
        self.trading_client = TradingClient(api_key, secret_key, paper=True)
        self.hist_client = StockHistoricalDataClient(api_key, secret_key)

    def get_account(self):
        # Fetch account details from Alpaca
        return self.trading_client.get_account()

    def place_order(self, symbol, qty, side, time_in_force="gtc"):
        # Place a market order for the specified symbol
        market_order_data = MarketOrderRequest(
                            symbol=symbol,
                            qty=qty,
                            side=side,
                            time_in_force=time_in_force
                            )

        market_order = self.trading_client.submit_order(
                        order_data=market_order_data
                    )
        
        return market_order

    def close_all_positions(self):
        # Close all open positions and cancel any pending orders
        self.trading_client.close_all_positions(cancel_orders=True)

    def get_live_price(self, symbol):
        # Fetch the latest trade price for the given symbol
        try:
            latest_trade = self.hist_client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))
            return latest_trade[symbol].price
        except Exception as e:
            print(f"Error fetching live price for {symbol}: {e}")
            return None
    
    def fetch_historical_data(self, ticker, start_date, end_date):
        # Fetch historical stock data for the given ticker and date range
        try:

            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )

            bars = self.hist_client.get_stock_bars(request)
            data = bars[ticker]
            df = pd.DataFrame([bar.__dict__ for bar in data])  

            df["timestamp"] = pd.to_datetime(df["timestamp"])  
            df.set_index("timestamp", inplace=True)  
            return df[["open", "high", "low", "close", "volume"]]  
            
        except Exception as e:
            print(f"Error fetching historical data for {ticker}: {e}")
            return None
        
    def get_current_allocations(self):
        # Calculate current portfolio allocations based on market value of positions
        try:
            positions = self.trading_client.get_all_positions()
            if not positions:
                return {}
          
            total_value = sum(float(pos.market_value) for pos in positions)
            if total_value == 0:
                return {pos.symbol: 0.0 for pos in positions}

            allocations = {
                pos.symbol: round(100 * float(pos.market_value) / total_value, 2)
                for pos in positions
            }
            return allocations
        except Exception as e:
            print(f"Error fetching current allocations: {e}")
            return {}