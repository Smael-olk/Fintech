import numpy as np
import pandas as pd
import scipy.stats as stats

def highlight_significant(val):
    """Highlights p-values less than 0.05 in bold red."""
    color = '#d62728' if val < 0.05 else 'inherit' # Red for significant
    weight = 'bold' if val < 0.05 else 'normal'
    return f'color: {color}; font-weight: {weight}'

def get_comprehensive_stats(series, freq='W'):
    annual_factor = 52 if freq == 'W' else 12
    cum_ret = (1 + series).cumprod()
    max_dd = -(1 - cum_ret / cum_ret.cummax()).max()

    return pd.Series({
        'Annualized Return': series.mean() * annual_factor,
        'Annualized Volatility': series.std() * np.sqrt(annual_factor),
        'Sharpe Ratio': (series.mean() * annual_factor) / (series.std() * np.sqrt(annual_factor)),
        'Max Drawdown': max_dd,
        'Skewness': series.skew(),
        'Kurtosis': series.kurtosis()
    })

def format_final_table(df):
    pct_rows = ['Annualized Return', 'Annualized Volatility', 'Max Drawdown']

    # Apply styling
    return df.style.format(lambda x: f"{x*100:.2f}%" if any(r in str(df.index) for r in pct_rows) else f"{x:.2f}")


def get_fama_french_data():
    """Fetches and cleans the Fama-French 3-Factor data."""
    import urllib.request, zipfile, io, ssl
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"

    context = ssl._create_unverified_context()
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    response = urllib.request.urlopen(req, context=context)

    with zipfile.ZipFile(io.BytesIO(response.read())) as z:
        with z.open(z.namelist()[0]) as f:
            lines = f.readlines()
            start = next(i for i, l in enumerate(lines) if b'Mkt-RF' in l)
            f.seek(0)
            df = pd.read_csv(f, skiprows=start, index_col=0)

    # Clean and convert to PeriodIndex
    df = df[df.index.astype(str).str.strip().str.match(r'^\d{6}$')]
    df.index = pd.to_datetime(df.index.astype(str).str.strip(), format='%Y%m').to_period('M')
    return df.astype(float) / 100


def calculate_var(returns, confidence=0.01, horizon=4, method='historical'):
    """
    Calculates Value at Risk (VaR) using robust methods.

    Parameters:
    returns (array-like): Series of returns
    confidence (float): Confidence level (e.g., 0.01 for 99%)
    horizon (int): Time horizon for scaling (e.g., 4 weeks)
    method (str): 'historical', 'modified' (Cornish-Fisher), or 'gaussian'
    """
    returns_array = np.array(returns)
    if len(returns_array) == 0:
        return np.nan

    if method == 'historical':
        # Pure non-parametric approach
        var_1w = -np.percentile(returns_array, confidence * 100)

    elif method == 'modified':
        # Cornish-Fisher expansion (Accounting for Skewness and Kurtosis found in EDA)
        z = stats.norm.ppf(confidence)
        s = stats.skew(returns_array)
        k = stats.kurtosis(returns_array)

        # Cornish-Fisher Adjusted Z-score
        z_cf = (z +
                (z ** 2 - 1) * s / 6 +
                (z ** 3 - 3 * z) * k / 24 -
                (2 * z ** 3 - 5 * z) * (s ** 2) / 36)

        var_1w = -(np.mean(returns_array) + z_cf * np.std(returns_array))

    else:  # Gaussian/Parametric
        z_score = stats.norm.ppf(confidence)
        var_1w = -(np.mean(returns_array) + z_score * np.std(returns_array))

    # Scale by square root of time
    var = var_1w * np.sqrt(horizon)
    return var

def var_t(returns_series, confidence=0.01, horizon=4):
    """
    Student-t parametric VaR.
    Fits degrees of freedom from data — better for fat-tailed returns.
    """
    if len(returns_series) < 10:  # Minimum safety check for distribution fitting
        return np.nan

    df, loc, scale = stats.t.fit(returns_series)
    q = stats.t.ppf(confidence, df=df, loc=loc, scale=scale)
    return -q * np.sqrt(horizon)
