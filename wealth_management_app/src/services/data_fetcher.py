import yfinance as yf


# DataFetcher class provides methods to fetch real-time and historical market data using yfinance
class DataFetcher:
    def fetch_market_data(self, ticker):
        """
        Fetch real-time market data for a given ticker symbol.
        Retrieves the latest market data using the yfinance library.
        """
        try:
            stock = yf.Ticker(ticker)  # Initialize yfinance Ticker object
            market_data = stock.history(period="1d")  # Fetch data for the last day
            latest_data = market_data.iloc[-1]  # Extract the latest data point
            return {
                "date": latest_data.name.strftime("%Y-%m-%d"),  # Format date
                "open": latest_data["Open"],
                "high": latest_data["High"],
                "low": latest_data["Low"],
                "close": latest_data["Close"],
                "volume": latest_data["Volume"],
            }
        except Exception as e:
            print(f"Error fetching market data for {ticker}: {e}")
            return None

    def fetch_historical_data(self, ticker, start_date, end_date):
        """
        Fetch historical market data for a given ticker symbol
        between specified start and end dates using the yfinance library.
        """
        try:
            stock = yf.Ticker(ticker)  # Initialize yfinance Ticker object
            historical_data = stock.history(start=start_date, end=end_date, auto_adjust=False)  # Fetch historical data

            if 'Adj Close' not in historical_data.columns:  # Check for adjusted close prices
                print(f"'Adj Close' not found for {ticker}")
                return None

            adjusted_close = historical_data[['Adj Close']].copy()  # Extract adjusted close prices
            adjusted_close.rename(columns={"Adj Close": "Close"}, inplace=True)  # Rename column for consistency

            adjusted_close.index = adjusted_close.index.date  # Convert index to date-only format

            return adjusted_close

        except Exception as e:
            print(f"Error fetching historical data for {ticker}: {e}")
            return None