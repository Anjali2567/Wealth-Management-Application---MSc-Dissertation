from matplotlib import pyplot as plt
from scipy import stats
from src.services.data_fetcher import DataFetcher
from services.macro_data_fetcher import MacroDataFetcher
from models.portfolio_optimization import PortfolioOptimization
from models.transformer import train_transformer
# from models.lstm import train_lstm
from sklearn.preprocessing import StandardScaler
from services.huggingface_service import HuggingFaceService
from services.config import FRED_API_KEY
import pandas as pd
import numpy as np


def predict_returns(all_tickers, start_date, end_date):
    """
    Predict returns using historical and macroeconomic data.
    :param tickers: List of selected tickers for prediction.
    :param all_tickers: List of all available tickers.
    :param start_date: Start date for historical data.
    :param end_date: End date for historical data.
    :return: Predicted returns and aligned data for optimization.
    """
    # Initialize services for macroeconomic data, trading, and historical data fetching
    macro_fetcher = MacroDataFetcher(FRED_API_KEY)
    data_fetcher = DataFetcher()

    # Fetch historical data for all tickers
    historical_data = {}
    for ticker in all_tickers:
        data = data_fetcher.fetch_historical_data(ticker, start_date, end_date)
        if data is not None:
            historical_data[ticker] = data["Close"]

    # Check if historical data is available
    if not historical_data:
        return None, None, "No historical data available for portfolio optimization."

    # Process historical data to calculate biweekly returns
    historical_data_df = pd.DataFrame(historical_data)
    historical_data_df.index = pd.to_datetime(historical_data_df.index)
    historical_data_df = historical_data_df.sort_index()  


    biweekly_prices = historical_data_df.resample('2W-FRI').last()
    returns_uncleaned = np.log(biweekly_prices / biweekly_prices.shift(1)).dropna()

    # Remove outliers and clip extreme values in returns
    z_scores = np.abs(stats.zscore(returns_uncleaned))
    returns = returns_uncleaned[(z_scores < 3).all(axis=1)]

    returns = returns.clip(lower=returns.quantile(0.01), upper=returns.quantile(0.99), axis=1)

    # Fetch macroeconomic indicators from FRED
    inflation_data = macro_fetcher.fetch_indicator("CPIAUCSL", start_date=start_date, end_date=end_date)
    gdp_data = macro_fetcher.fetch_indicator("GDPC1", start_date=start_date, end_date=end_date)
    federal_data = macro_fetcher.fetch_indicator("FEDFUNDS", start_date=start_date, end_date=end_date)
    unemployment_data = macro_fetcher.fetch_indicator("UNRATE", start_date=start_date, end_date=end_date)
    treasury_data = macro_fetcher.fetch_indicator("GS10", start_date=start_date, end_date=end_date)
    macro_data_df = pd.DataFrame({
        "Inflation": inflation_data,
        "GDP": gdp_data,
        "Federal Funds Rate": federal_data,
        "Unemployment Rate": unemployment_data,
        "10-Year Treasury Yield": treasury_data
    }).dropna()

    # Resample and align macroeconomic data with returns
    macro_data_df = macro_data_df.resample('2W-FRI').ffill().dropna()

    if not macro_data_df.empty:
        if not isinstance(macro_data_df.index, pd.DatetimeIndex):
            macro_data_df.index = pd.to_datetime(macro_data_df.index)
        if macro_data_df.index.tz is not None:
            macro_data_df.index = macro_data_df.index.tz_localize(None)
    if not isinstance(returns.index, pd.DatetimeIndex):
        returns.index = pd.to_datetime(returns.index)
    if getattr(returns.index, "tz", None) is not None:
        returns.index = returns.index.tz_localize(None)
    if not macro_data_df.empty:
        macro_data_df = macro_data_df.reindex(returns.index, method="ffill").fillna(method="ffill")

    # Scale returns and macroeconomic data
    returns_scaler = StandardScaler()
    macro_scaler = StandardScaler()
    scaled_returns = pd.DataFrame(
        returns_scaler.fit_transform(returns),
        index=returns.index,
        columns=returns.columns
    )
    scaled_macro = pd.DataFrame(
        macro_scaler.fit_transform(macro_data_df),
        index=macro_data_df.index,
        columns=macro_data_df.columns
    )

    # Calculate moving averages and RSI for uncleaned return data
    moving_averages = {}
    rsi_values = {}
    for ticker in returns_uncleaned.columns:
        returns_series = returns_uncleaned[ticker]
        moving_averages[ticker] = {
            "MA_12": returns_series.rolling(window=12).mean(),
            "MA_24": returns_series.rolling(window=24).mean()
        }
        delta = returns_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi_values[ticker] = 100 - (100 / (1 + rs))

    # Combine moving averages and RSI into a DataFrame
    ma_rsi_data = pd.concat({
        ticker: pd.DataFrame({**moving_averages[ticker], "RSI": rsi_values[ticker]})
        for ticker in returns_uncleaned.columns
    }, axis=1)

    # Align moving averages and RSI with returns
    ma_rsi_data = ma_rsi_data.reindex(returns.index, method="ffill").dropna()

    # Combine scaled returns, macroeconomic data, and technical indicators
    combined_data = pd.concat([scaled_returns, scaled_macro, ma_rsi_data], axis=1).dropna()
    asset_columns = list(returns.columns)

    # Prepare features and targets for model training
    target_scaled = scaled_returns[asset_columns].shift(-1)
    aligned = combined_data.join(target_scaled, how='inner', rsuffix='_target').dropna()

    feature_cols = combined_data.columns
    target_cols = asset_columns

    feat_array = aligned[feature_cols].values.astype(np.float32)
    targ_array = aligned[target_cols].values.astype(np.float32)

    # Train transformer model and predict next returns
    next_scaled_returns = train_transformer(feat_array, targ_array, returns_scaler)

    predicted_unscaled = returns_scaler.inverse_transform(next_scaled_returns.reshape(1, -1)).ravel()
    predicted_returns = pd.Series(predicted_unscaled, index=asset_columns)

    return predicted_returns, returns

def optimize_portfolio(tickers, predicted_returns, returns, risk_pref):
    """
    Optimize portfolio using predicted returns and historical returns.
    :param tickers: List of selected tickers for optimization.
    :param predicted_returns: Predicted returns for the selected tickers.
    :param returns: Historical returns for the selected tickers.
    :param risk_pref: User's risk preference for optimization.
    :return: Optimized portfolio weights.
    """
    # Perform portfolio optimization using predicted returns
    returns_for_optimization = returns[tickers]
    predicted_returns_for_optimization = predicted_returns[tickers]
    asset_columns = list(returns.columns)

    portfolio_optimizer = PortfolioOptimization(returns_for_optimization, predicted_returns_for_optimization)
    optimized_weights = portfolio_optimizer.optimize_portfolio(risk_profile=risk_pref)
    if optimized_weights is None:
        return None

    optimized_weights = [max(0, weight) for weight in optimized_weights]
    total_weight = sum(optimized_weights)
    percentage_weights = {ticker: round((weight / total_weight) * 100, 2) for ticker, weight in zip(asset_columns, optimized_weights)}

    percentage_weights = {ticker: weight for ticker, weight in percentage_weights.items() if weight > 0.00}

    # Calculate portfolio metrics (Sharpe and Sortino ratios)
    risk_free_return = 0.000769
    sharpe_ratio = portfolio_optimizer.calculate_sharpe_ratio(optimized_weights, risk_free_return)
    sortino_ratio = portfolio_optimizer.calculate_sortino_ratio(optimized_weights, risk_free_return)

    print("Sharpe Ratio:", sharpe_ratio)
    print("Sortino Ratio:", sortino_ratio)

    hf_service = HuggingFaceService()
    # Interpret portfolio results using HuggingFaceService
    explanation = hf_service.interpret_portfolio_results(
        weights=optimized_weights,
        tickers=asset_columns,
        predicted_returns=predicted_returns
    )

    # Return optimized weights and predicted returns
    return percentage_weights, explanation