import pandas as pd

def consolidate_one_hot_columns(df, prefix, drop_original=False):
    """
    Consolidates one-hot encoded columns with a given prefix back into a single categorical column.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the one-hot encoded columns.
    - prefix (str): The prefix used in the one-hot encoded columns that should be consolidated.

    Returns:
    - The DataFrame with the one-hot encoded columns consolidated into a single column.
    """
    columns = [col for col in df.columns if col.startswith(prefix)]
    df[prefix.rstrip('_')] = df[columns].idxmax(axis=1).str.replace(prefix, '')
    
    if drop_original:
        df.drop(columns, axis=1, inplace=True)

    return df

def convert_binary_to_boolean(df, column_name: str, drop_original: bool = False) -> pd.DataFrame:
    """
    Convert binary values (0, 1) in a DataFrame column to boolean strings ('false', 'true').
    If the column name ends with '_yes', it is stripped off. The original column can be dropped.

    Parameters:
    df (pd.DataFrame): DataFrame to modify
    column_name (str): Name of the column to convert
    drop_original (bool): Whether to drop the original column after conversion

    Returns:
    pd.DataFrame: Modified DataFrame with converted column. None if an error occurs.
    """
    if column_name not in df.columns:
        print(f"Error: The column '{column_name}' does not exist in the DataFrame.")
        return None

    unique_values = df[column_name].unique()
    if not set(unique_values).issubset({0, 1}):
        print(f"Error: The column '{column_name}' contains values other than 0 and 1.")
        return None

    new_column_name = column_name.rstrip('_yes') if column_name.endswith('_yes') else column_name
    df[new_column_name] = df[column_name].map({0: 'false', 1: 'true'})

    if drop_original and new_column_name != column_name:
        df.drop(columns=[column_name], inplace=True)

    return df