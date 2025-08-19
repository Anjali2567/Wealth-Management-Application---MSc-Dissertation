from fredapi import Fred

# MacroDataFetcher class provides methods to fetch macroeconomic data from FRED
class MacroDataFetcher:
    def __init__(self, api_key):
        """
        Initialize the MacroDataFetcher class with the FRED API key.
        :param api_key: Your FRED API key.
        """
        self.fred = Fred(api_key=api_key)  # Initialize FRED API client

    def fetch_indicator(self, indicator_code, start_date=None, end_date=None):
        """
        Fetch macroeconomic indicator data from FRED.
        :param indicator_code: The FRED code for the indicator (e.g., 'CPIAUCSL' for inflation).
        :param start_date: Start date for the data (optional).
        :param end_date: End date for the data (optional).
        :return: A pandas Series containing the indicator data within the specified date range.
        """
        try:
            data = self.fred.get_series(indicator_code)  
            if start_date and end_date:
                data = data.loc[start_date:end_date] 
            return data
        except Exception as e:
            print(f"Error fetching data for indicator {indicator_code}: {e}")
            return None