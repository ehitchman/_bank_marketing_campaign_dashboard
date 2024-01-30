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
    model_predictions_bank_csv_filepath = os.path.join('.', 'data', 'bank_model predictions.csv')

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

    # Run the data modeler
    prediction_probabilities = data_modeler.run()

    # Consolidate one-hot encoded columns
    prefixes = ['age_', 'job_', 'marital_', 'education_']
    for prefix in prefixes:
        prediction_probabilities = utilities.consolidate_one_hot_columns(
            prediction_probabilities, 
            prefix, 
            drop_original=True
            )

    # Convert binary values (0, 1) to boolean strings ('false', 'true')
    columns = ['default_yes', 'housing_yes', 'loan_yes']
    for column in columns:
        prediction_probabilities = utilities.convert_binary_to_boolean(
            prediction_probabilities, 
            column,
            drop_original=True
            )
    print('\n----------------------------')
    print('data prepared for export'.center(28, '-'))
    print(type(prediction_probabilities))
    print(prediction_probabilities.head(5))

    # Export data
    print('\n----------------------------')
    print('EXPORT DATA'.center(28, '-'))
    data_loader.export_data(
        prediction_probabilities, 
        model_predictions_bank_csv_filepath
        )

if __name__ == "__main__":
    data_loader = main()
