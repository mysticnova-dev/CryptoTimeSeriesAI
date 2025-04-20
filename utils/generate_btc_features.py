import pandas as pd
import numpy as np

     
def generate_volatility_indicators(df, price_col='btc_market_price', window=20):
    """
    Generate volatility indicators including:
    - Bollinger Bands (Upper, Lower, Middle)
    - Average True Range (ATR)
    - High-Low Difference and % Difference

    Parameters:
        df (pd.DataFrame): Input DataFrame (time-ordered).
        price_col (str): Name of the price column.
        window (int): Rolling window for Bollinger and ATR.

    Returns:
        pd.DataFrame: DataFrame with volatility indicators added.
    """
    df = df.copy()

    # --- Bollinger Bands ---
    rolling_mean = df[price_col].rolling(window=window, min_periods=1).mean()
    rolling_std = df[price_col].rolling(window=window, min_periods=1).std()
    
    df['bollinger_middle'] = rolling_mean
    df['bollinger_upper'] = rolling_mean + 2 * rolling_std
    df['bollinger_lower'] = rolling_mean - 2 * rolling_std

    # --- Average True Range (ATR) approximation using price range ---
    df['atr'] = df[price_col].rolling(window=14, min_periods=1).apply(
        lambda x: x.max() - x.min(), raw=True
    )

    # --- High-Low Difference (24h-style) ---
    df['high_low_diff'] = df[price_col].rolling(window=1).apply(
        lambda x: x.max() - x.min(), raw=True
    )

    # --- High-Low % Difference ---
    df['high_low_pct_diff'] = (df['high_low_diff'] / df[price_col]) * 100

    return df
    
    
def generate_momentum_indicators(df, price_col='btc_market_price', momentum_period=10):
    """
    Generate momentum and rate-of-change (ROC) indicators.

    Indicators Created:
        - Momentum (MOM): Price - Price.shift(momentum_period)
        - Rate of Change (ROC): ((Price - Price.shift(momentum_period)) / Price.shift(momentum_period)) * 100

    Parameters:
        df (pd.DataFrame): Input DataFrame (time-ordered).
        price_col (str): Name of the price column.
        momentum_period (int): Number of periods to calculate momentum over.

    Returns:
        pd.DataFrame: DataFrame with momentum indicators added.
    """
    df = df.copy()
    shifted = df[price_col].shift(momentum_period)
    
    df['momentum'] = df[price_col] - shifted
    df['roc'] = ((df[price_col] - shifted) / shifted) * 100
    
    return df

def generate_volume_indicators(df, price_col='btc_market_price', volume_col='btc_trade_volume'):
    """
    Generate volume-based indicators:
    - RVOL: Relative Volume (volume / volume rolling mean)
    - VPT: Volume Price Trend
    - OBV: On-Balance Volume

    Parameters:
        df (pd.DataFrame): Input DataFrame (time-ordered).
        price_col (str): Name of the price column.
        volume_col (str): Name of the volume column.

    Returns:
        pd.DataFrame: DataFrame with volume indicators added.
    """
    df = df.copy()

    # --- Relative Volume (RVOL) ---
    df['rvol'] = df[volume_col] / df[volume_col].rolling(window=30, min_periods=1).mean()

    # --- Volume Price Trend (VPT) ---
    price_change_pct = (df[price_col] - df[price_col].shift(1)) / (df[price_col].shift(1) + 1e-9)
    df['vpt'] = (df[volume_col] * price_change_pct).cumsum()

    # --- On-Balance Volume (OBV) ---
    direction = np.sign(df[price_col].diff()).fillna(0)
    df['obv'] = (direction * df[volume_col]).fillna(0).cumsum()

    return df


def generate_market_strength_indicators(df):
    """
    Generate supply and strength indicators:
    - circulating_to_total_supply_ratio
    - market_cap_to_circulating_supply
    - price_to_circulating_supply
    - price_to_total_supply

    Assumes:
        - btc_total_bitcoins = circulating supply
        - market cap and price already exist

    Returns:
        pd.DataFrame: DataFrame with market strength indicators added.
    """
    df = df.copy()
    eps = 1e-9

    # Circulating Supply / Total Supply
    # Assuming maximum Bitcoin supply = 21 million
    df['circulating_to_total_supply_ratio'] = df['btc_total_bitcoins'] / (21_000_000 + eps)

    # Market Cap / Circulating Supply
    df['market_cap_to_circulating_supply'] = df['btc_market_cap'] / (df['btc_total_bitcoins'] + eps)

    # Price / Circulating Supply
    df['price_to_circulating_supply'] = df['btc_market_price'] / (df['btc_total_bitcoins'] + eps)

    # Price / Total Supply (constant theoretical max)
    df['price_to_total_supply'] = df['btc_market_price'] / (21_000_000 + eps)

    return df
    
    
def generate_combined_ratios(df, price_col='btc_market_price', volume_col='btc_trade_volume'):
    """
    Generate combined ratio indicators:
    - Price to Volume Ratio
    - Volume(24h) to Volume(7d)
    - Volume(7d) to Volume(30d)
    - Volume(24h) to Volume(30d)

    Assumes volume column is daily and 7d/30d volumes are computed via rolling sums.

    Returns:
        pd.DataFrame: DataFrame with combined ratio features added.
    """
    df = df.copy()
    eps = 1e-9

    # Rolling volume aggregates
    df['volume_7d'] = df[volume_col].rolling(window=7, min_periods=1).sum()
    df['volume_30d'] = df[volume_col].rolling(window=30, min_periods=1).sum()

    # Ratios
    df['price_to_volume_ratio'] = df[price_col] / (df[volume_col] + eps)
    df['volume_24_to_7d'] = df[volume_col] / (df['volume_7d'] + eps)
    df['volume_7d_to_30d'] = df['volume_7d'] / (df['volume_30d'] + eps)
    df['volume_24_to_30d'] = df[volume_col] / (df['volume_30d'] + eps)

    return df
    
    
def generate_log_transforms(df, columns):
    """
    Generate log-transformed versions of specified columns.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of column names to log-transform.

    Returns:
        pd.DataFrame: DataFrame with log-transformed features added.
    """
    df = df.copy()
    for col in columns:
        log_col = f'log_{col}'
        df[log_col] = np.log(df[col].replace(0, np.nan)).replace(np.nan, 0)
    return df
    
    
class BTCFeatureEngineer:
    def __init__(self, price_col='btc_market_price', volume_col='btc_trade_volume'):
        self.price_col = price_col
        self.volume_col = volume_col

    def transform(self, df):
        df = df.copy()
        df = generate_volatility_indicators(df, self.price_col)
        df = generate_momentum_indicators(df, self.price_col)
        df = generate_volume_indicators(df, self.price_col, self.volume_col)
        df = generate_market_strength_indicators(df)
        df = generate_combined_ratios(df, self.price_col, self.volume_col)

        log_cols = [
            'btc_market_price', 'btc_market_cap', 'btc_total_bitcoins',
            'btc_trade_volume', 'btc_output_volume',
            'btc_estimated_transaction_volume', 'btc_estimated_transaction_volume_usd',
            'btc_transaction_fees', 'btc_cost_per_transaction', 'btc_miners_revenue'
        ]
        df = generate_log_transforms(df, log_cols)

        # Fill NaNs with -101010 and add _missing flag columns
        df.fillna(-101010, inplace=True)
        for col in df.columns:
            if df[col].dtype != 'object':
                df[f'{col}_missing'] = df[col] == -101010

        return df

