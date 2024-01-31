import os

from classes.DataHandler import DataHandler
from classes.DataModeler import DataModeler

from classes import utilities

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

    print('\n----------------------------')
    print('EXPORT TEST SPLIT PREDICTIONS'.center(28, '-'))
    # Consolidate one-hot encoded columns
    prefixes = ['age_', 'job_', 'marital_', 'education_']
    for prefix in prefixes:
        data_modeler.x_test = utilities.consolidate_one_hot_columns(
            data_modeler.x_test, 
            prefix, 
            drop_original=True
            )

    # Convert binary values (0, 1) to boolean strings ('false', 'true')
    columns = ['default_yes', 'housing_yes', 'loan_yes']
    for column in columns:
        data_modeler.x_test = utilities.convert_binary_to_boolean(
            data_modeler.x_test, 
            column,
            drop_original=True
            )
        
    print('test split predictions export ready'.center(28, '-'))
    print(type(data_modeler.x_test))
    print(data_modeler.x_test.head(5))
    data_loader.export_data(
        data_modeler.x_test, 
        model_testsplit_predictions_csv_filepath
        )

    print('\n----------------------------')
    print('EXPORT MODEL ALL PREDICTIONS'.center(28, '-'))
    # Consolidate one-hot encoded columns

    print(data_modeler.all_processed_data_with_predictions.dtypes)

    prefixes = ['age_', 'job_', 'marital_', 'education_']
    for prefix in prefixes:
        data_modeler.all_processed_data_with_predictions = utilities.consolidate_one_hot_columns(
            data_modeler.all_processed_data_with_predictions, 
            prefix, 
            drop_original=True
            )

    # Convert binary values (0, 1) to boolean strings ('false', 'true')
    columns = ['default_yes', 'housing_yes', 'loan_yes']
    for column in columns:
        data_modeler.all_processed_data_with_predictions = utilities.convert_binary_to_boolean(
            data_modeler.all_processed_data_with_predictions, 
            column,
            drop_original=True
            )
        
    print('all predictions export ready'.center(28, '-'))
    print(type(data_modeler.all_processed_data_with_predictions))
    print(data_modeler.all_processed_data_with_predictions.head(75))
    data_loader.export_data(
        data_modeler.all_processed_data_with_predictions, 
        model_all_predictions_csv_filepath
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