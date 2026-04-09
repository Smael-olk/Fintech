import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def prepare_features(df):
    X = df.copy()

    # Strip trailing spaces from column names upfront
    X.columns = X.columns.str.strip()

    # Engineered features
    X['IncomePerFamilyMember'] = X['Income'] / X['FamilyMembers'].replace(0, 1)
    X['WealthToIncome'] = np.log1p(X['Wealth'].div(X['Income'].replace(0, np.nan)).fillna(0))
    X['SophisticationScore'] = X['FinancialEducation'] * X['RiskPropensity']

    # Cast Gender as categorical so FAMD treats it correctly
    X['Gender'] = X['Gender'].astype('category')

    features = [
        'Age', 'Gender', 'FamilyMembers', 'FinancialEducation',
        'RiskPropensity', 'Income', 'Wealth',
        'IncomePerFamilyMember', 'WealthToIncome',
        'SophisticationScore'
    ]

    return X[features]




def create_variable_summary(df, metadata_df):
    # Create empty lists to store the chosen statistics
    stats_dict = {
        'Variable': [],
        'Description': [],
        'Mean': [],
        'Std': [],
        'Missing': [],
        'Min': [],
        'Max': []
    }

    # Create a metadata dictionary for easy lookup
    meta_dict = dict(zip(metadata_df['Metadata'], metadata_df['Unnamed: 1']))

    for col in df.columns:
        stats_dict['Variable'].append(col)
        stats_dict['Description'].append(meta_dict.get(col, 'N/A'))

        # Calculate some statistics for each column
        if pd.api.types.is_numeric_dtype(df[col]):
            stats_dict['Mean'].append(f"{df[col].mean():.2f}")
            stats_dict['Std'].append(f"{df[col].std():.2f}")
            stats_dict['Min'].append(f"{df[col].min():.2f}")
            stats_dict['Max'].append(f"{df[col].max():.2f}")
        else:
            stats_dict['Mean'].append('N/A')
            stats_dict['Std'].append('N/A')
            stats_dict['Min'].append('N/A')
            stats_dict['Max'].append('N/A')

        stats_dict['Missing'].append(df[col].isna().sum())

    return pd.DataFrame(stats_dict)
