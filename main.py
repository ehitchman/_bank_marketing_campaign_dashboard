import os

from classes.DataHandler import DataHandler
from classes.DataModeler import DataModeler

from classes import utilities

def process_predictions_data(data, prefixes, columns):
    """
    Consolidates one-hot encoded columns and converts binary values to boolean strings.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the one-hot encoded columns.
    - prefixes (list): The prefixes used in the one-hot encoded columns that should be consolidated.
    - columns (list): The columns that should be converted from binary to boolean.

    Returns:
    - The DataFrame with the one-hot encoded columns consolidated into a single column and binary values converted to boolean strings.
    """
    for prefix in prefixes:
        data = utilities.consolidate_one_hot_columns(
            data, 
            prefix, 
            drop_original=True
            )

    for column in columns:
        data = utilities.convert_binary_to_boolean(
            data, 
            column,
            drop_original=True
            )

    return data

def main():
    """
    This function loads the raw data, runs the data modeler, and exports the model output data to a csv file.
    """    
    # Set file paths
    raw_bank_csv_filepath = os.path.join('.', 'data', 'bank.csv')
    model_testsplit_predictions_csv_filepath = os.path.join('.', 'data', 'model_output', 'bank_model testing split predictions.csv')
    model_coefficients_csv_filepath = os.path.join('.', 'data', 'model_output', 'bank_model coefficients.csv')
    model_all_predictions_csv_filepath = os.path.join('.', 'data', 'model_output', 'bank_predictions on all data.csv')

    # Create a data loader object and load data
    data_loader = DataHandler(filepath=raw_bank_csv_filepath)
    data = data_loader.data

    # Create an instance of data modeler
    numeric_vars=['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'],
    categorical_vars=['job', 'marital', 'education', 'default', 'housing', 'loan']
    data_modeler = DataModeler(
        data=data, 
        target_variable='deposit', 
        numeric_vars=numeric_vars,
        categorical_vars=categorical_vars
        )

    # Run the data modeler (outputs x_test predictions, coefficients, odds ratios)
    data_modeler.run()

    # Consolidate one-hot encoded columns and convert binary values to boolean strings
    prefixes = ['age_', 'job_', 'marital_', 'education_']
    columns = ['default_yes', 'housing_yes', 'loan_yes'] 
    dfs_config = {
        'x_test': model_testsplit_predictions_csv_filepath, 
        'all_processed_data_with_predictions': model_all_predictions_csv_filepath
        }
    
    # Consolidate one-hot encoded columns, convert binary values to boolean, 
    # and export predictions
    print('\n----------------------------')
    print('EXPORT PREDICTIONS'.center(28, '-'))
    for df_name, export_filepath in dfs_config.items():
        df = getattr(data_modeler, df_name)
        processed_df = process_predictions_data(df, prefixes, columns)
        setattr(data_modeler, df_name, processed_df)

        data_loader.export_data(
            processed_df, 
            export_filepath
            )

    print('\n----------------------------')
    print('EXPORT COEFFICIENTS'.center(28, '-'))
    data_loader.export_data(
        data_modeler.coeff_summary, 
        model_coefficients_csv_filepath
        )
    
    return data_modeler

if __name__ == "__main__":
    data_modeler = main()
    print(data_modeler.all_processed_data_with_predictions.sort_values(by='deposit', ascending=True).head(75))