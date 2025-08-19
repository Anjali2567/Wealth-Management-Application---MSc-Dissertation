import datetime
from .alpaca_trading import AlpacaTrading
from .config import ALPACA_API_KEY, ALPACA_SECRET_KEY

# Initialize AlpacaTrading instance with API keys
alpaca_trading = AlpacaTrading(api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY)

def execute_trades(weights):
    """
    Executes trades in Alpaca based on the given weights dictionary.
    :param weights: dict of {ticker: allocation_percentage}
    :return: List of trade results (messages)
    """
    messages = [] 
    account = alpaca_trading.get_account()  # Fetch account details

    total_cash = float(account.cash)  # Get available cash in the account
    for ticker, weight in weights.items():
        allocation = total_cash * (weight / 100)  
        current_price = alpaca_trading.get_live_price(ticker)  # Fetch live price of the ticker
        if current_price is None:
            messages.append(f"Could not fetch live price for {ticker}. Skipping...")
            continue
        qty = int(allocation // current_price)  
        if qty > 0:
            alpaca_trading.place_order(symbol=ticker, qty=qty, side="buy") 
            messages.append(f"Placed order for {qty} shares of {ticker}")
        else:
            messages.append(f"Allocation for {ticker} too small to buy any shares.")
    return messages

def check_rebalance_needed(target_allocations, threshold=5.0):
    """
    Check if portfolio rebalancing is needed based on target allocations and threshold.
    :param target_allocations: dict of {ticker: target_allocation_percentage}
    :param threshold: Percentage difference threshold to trigger rebalancing.
    :return: Tuple (rebalance_needed, rebalance_assets)
    """
    rebalance_needed = False  
    rebalance_assets = {}  
    current_allocations = alpaca_trading.get_current_allocations()  # Fetch current portfolio allocations
    for asset, target_weight in target_allocations.items():
        current_weight = current_allocations.get(asset, 0.0) 
        diff = abs(current_weight - target_weight)  
        if diff > threshold:
            rebalance_needed = True
            rebalance_assets[asset] = {
                "Target (%)": target_weight,
                "Current (%)": current_weight,
                "Difference (%)": round(diff, 2)
            }
    return rebalance_needed, rebalance_assets

def get_last_working_day():
    """
    Get the last working day (Monday to Friday).
    :return: Date string of the last working day.
    """
    today = datetime.date.today()  
    offset = 1  
    while True:
        candidate = today - datetime.timedelta(days=offset)  
        if candidate.weekday() < 5:  
            return candidate.strftime("%Y-%m-%d")
        offset += 1

def get_asset_names_from_alpaca(tickers):
    """
    Fetch asset names for given tickers from Alpaca.
    :param tickers: List of ticker symbols.
    :return: Dictionary of {ticker: asset_name}
    """
    asset_names = {} 
    for ticker in tickers:
        try:
            asset_info = alpaca_trading.trading_client.get_asset(ticker)
            asset_names[ticker] = asset_info.name if hasattr(asset_info, "name") else ticker
        except Exception as e:
            asset_names[ticker] = ticker
    return asset_names