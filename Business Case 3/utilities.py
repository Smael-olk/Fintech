import numpy as np
import pandas as pd


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


