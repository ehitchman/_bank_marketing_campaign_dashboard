import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer

class DataModeler:
    def __init__(self, data, target_variable: str, numeric_columns: list, categorical_columns: list, test_size: float = 0.3, random_state: int = 42):
        
        # Initialize data
        self.data = data
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.target_variable = target_variable

        # Initialize data preprocessing variables
        self.encoder = OneHotEncoder(sparse_output=False, drop='first')
        self.test_size = test_size
        self.random_state = random_state

        # Initialize model
        self.model = LogisticRegression(max_iter=10000)

        # Initialize data splits
        self.features_train = None
        self.features_test = None
        self.y_train = None
        self.y_test = None

    def _finalize_data(self):
        """
        Retains only the specified numeric variables, target variable, and encoded categorical variables in the processed data.
        """
        #TODO: Could potentially be automated using inclusion of 'continuous_vars' argument in function signature and
        #     checking if the variable is in the list of any variables (requires some review)
        # all_vars = self.numeric_columns + [self.target_variable] + list(self.encoder.get_feature_names_out(self.categorical_columns))
        # self.processed_data = self.processed_data[all_vars]
        
        #temporary solution to remove variables that shouldn't be included inthe model
        extraneous_vars = ['id', 'contact', 'day', 'month']
        self.processed_data = self.processed_data.drop(extraneous_vars, axis=1)

    def preprocess_data(self):
        """
        Encode categorical variables using one-hot encoding and prepares data set for training

        NOTE: The columns relied on in this function are hard-coded and will need to be updated if the data set changes.
        """
        # Create a copy of the data in a new variable
        self.processed_data = self.data.copy()

        print('\n----------------------------')
        print('PREPROCESS DATA'.center(28, '-'))
        print('----------------------------')

        print('\n----------------------------')
        print('raw data preview and data types'.center(28, '-'))
        print(self.processed_data.dtypes)
        print(self.processed_data.head(5))

        # # Encode the number of banking products by counting the number of 'yes' values in the banking products columns
        # columns = ['default', 'housing', 'loan']
        # self.processed_data['Count of Banking Products'] = self.processed_data[columns].apply(lambda x: x.eq('yes').sum())
                
        # Encode poutcome using a binary variable based on whether the previous campaign was successful or not ('Previous
        # Campaign Outcome')
        column_name = 'poutcome'
        new_column_name = column_name+'_binary'
        self.processed_data[new_column_name] = self.processed_data[column_name].apply(lambda x: 1 if x == 'success' else 0)
        self.processed_data = self.processed_data.drop(column_name, axis=1)

        # Encode age with custom bucket sizes (note, we also use one-hot encoding here separately as its not a part of
        # the initial data set and we don't necessarily think that age is a linear relationship with the target variable)
        column_name = 'age'
        new_column_name = 'age_'+column_name
        bins = [18, 30, 40, 50, 60, 150]
        bin_labels = ['18-29', '30-39', '40-49', '50-59', '60+']
        self.processed_data[new_column_name] = pd.cut(self.processed_data[column_name], bins=bins, labels=bin_labels, right=False)
        self.processed_data = self.processed_data.drop(column_name, axis=1)
        self.processed_data = pd.get_dummies(self.processed_data, columns=[new_column_name], drop_first=False)
        
        # Convert age columns to int
        age_columns = [col for col in self.processed_data.columns if 'age_' in col]
        self.processed_data[age_columns] = self.processed_data[age_columns].astype(int)
 
        # Encode balance with binned values
        # TODO: Add as a conditional based on an argument 'continuous_vars: list = ['balance']' function signature
        # if self.continuous_vars:
        column_name = 'balance'
        new_column_name = 'binned_'+column_name
        est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        self.processed_data[new_column_name] = est.fit_transform(self.processed_data[[column_name]])
        self.processed_data = self.processed_data.drop(column_name, axis=1)

        # Encode campaign (in-campaign contacts)
        # TODO: Add as a conditional based on a argument 'capped_vars: list = ['balance']' function signature
        column_name = 'campaign'
        new_column_name = column_name+'_capped'
        threshold = self.processed_data[column_name].quantile(0.99)
        self.processed_data[new_column_name] = self.processed_data[column_name].apply(lambda x: min(x, threshold))
        self.processed_data.drop(column_name, axis=1, inplace=True)

        # Encode previous (in-campaign contacts) using a binary variable based on whether they have been contacted or not
        # TODO: Add as a conditional based on a argument 'capped_vars: list = ['balance']' function signature
        column_name = 'previous'
        new_column_name = column_name+'_binary'
        self.processed_data[new_column_name] = self.processed_data[column_name].apply(lambda x: 0 if x == -1 else 1)
        self.processed_data = self.processed_data.drop(column_name, axis=1)

        # Encode pdays using a binary variable based on whether they had been contacted or not in the previous campaign
        # contacted from previous campaign ('Days Since Last Contacted')
        column_name = 'pdays'
        new_column_name = column_name+'_binary'
        self.processed_data[new_column_name] = self.processed_data[column_name].apply(lambda x: 0 if x == -1 else 1)
        self.processed_data = self.processed_data.drop(column_name, axis=1)

        # Encode all listed categorical variables using one-hot encoding
        if self.categorical_columns:
            encoded_vars = self.encoder.fit_transform(self.processed_data[self.categorical_columns])
            encoded_df = pd.DataFrame(encoded_vars, columns=self.encoder.get_feature_names_out(self.categorical_columns))
            print(encoded_df.head(5))
            self.processed_data = pd.concat([self.processed_data.drop(self.categorical_columns, axis=1), encoded_df], axis=1)

        # Encode target variable using one-hot encoding
        if self.processed_data[self.target_variable].dtype == 'object':
            self.processed_data[self.target_variable] = self.processed_data[self.target_variable].map({'yes': 1, 'no': 0})

        # Keep only the selected numeric variables, target, and encoded categorical variables
        self._finalize_data()

        print('\n----------------------------')
        print('model input data preview/types'.center(28, '-'))
        print(self.processed_data.head(5))
        print(self.processed_data.dtypes)
    
    def split_data(self):
        """
        Split data into training and testing sets
        """
        features = self.processed_data.drop(self.target_variable, axis=1)
        target = self.processed_data[self.target_variable]
        self.features_train, self.features_test, self.y_train, self.y_test = train_test_split(features, target, test_size=self.test_size, random_state=self.random_state)
        
        print('\n----------------------------')
        print('SPLIT TRAINING/TESTING'.center(28, '-'))
        print('----------------------------')

        print('----------------------------')
        print('self.features_train'.center(28, '-'))
        print(self.features_train.head(5))

        print('\n----------------------------')
        print('self.features_test'.center(28, '-'))
        print(self.features_test.head(5))

        print('\n----------------------------')
        print('self.y_train'.center(28, '-'))
        print(self.y_train.head(5))

        print('\n----------------------------')
        print('self.y_test'.center(28, '-'))
        print(self.y_test.head(5))

    def train_and_evaluate_model(self):
        """
        Trains and evaluates the model using the testing dataset and prints the 
        classification report and confusion matrix.
        """
        self.model.fit(self.features_train, self.y_train)
        self.predictions = self.model.predict(self.features_test)
        print('----------------------------')
        
        print('\n----------------------------')
        print('TRAIN AND EVALUATE MODEL'.center(28, '-'))

        print('\n----------------------------')
        print('classification report'.center(28, '-'))
        print(classification_report(self.y_test, self.predictions))
        
        print('\n----------------------------')
        print('confusion matrix'.center(28, '-'))
        print(confusion_matrix(self.y_test, self.predictions))

    def extract_prediction_probabilities(self):
        """
        Extracts the prediction probabilities for the test data set and attaches 
        them to the data set.
        """
        print('\n----------------------------')
        print('EXTRACT PREDICTIONS'.center(28, '-'))
        print('----------------------------')
        
        self.test_predictions_probabilities = self.model.predict_proba(self.features_test)
        self.features_test['test_predictions_probabilities'] = self.test_predictions_probabilities[:, 1] 
        print('\n----------------------------')
        print('attached probabilities'.center(28, '-'))
        print(self.features_test.head(5))

    def apply_model_to_processed_data(self):
        """
        Applies the trained logistic regression model to the processed data stored in
        self.processed_data, predicts probabilities, and attaches these predictions as
        a new column to a copy of the processed data.
        
        Returns:
        - data_with_predictions: a copy of self.processed_data with an additional
        column for predicted probabilities.
        """
        # Ensure the model is trained
        if self.model:
            data_with_predictions = self.processed_data.copy()

            # Predict probabilities for the processed data
            probabilities = self.model.predict_proba(data_with_predictions.drop('deposit', axis=1))
            positive_class_probabilities = probabilities[:, 1]

            # Attach probabilities to the copied dataset
            data_with_predictions['test_predictions_probabilities'] = positive_class_probabilities

            self.data_with_predictions = data_with_predictions
            print("Model predictions attached to a copy of self.processed_data.")
            return None

        else:
            print("Model not trained. Please train the model before applying it to the processed data.")
            return None

    def extract_coefficients_and_odds_ratios(self):
        """
        Extracts the coefficients and odds ratios from the logistic regression model
        and creates a summary DataFrame.
        """
        if self.model:
            # Extract coefficients/odds_ratios from the model
            coefficients = self.model.coef_[0]
            odds_ratios = np.exp(coefficients)
            
            # Ensure feature names match the trained model's features
            feature_names = self.features_train.columns
            
            # Create a summary DataFrame
            coeff_summary = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefficients,
                'Odds Ratio': odds_ratios
            }).sort_values(by='Odds Ratio', ascending=False)
            
            print("\nCoefficients and Odds Ratios Summary:")
            print(coeff_summary.head(5))
            
            # Add to the class instance
            self.coeff_summary = coeff_summary.sort_values(by='Feature', ascending=True)

        else:
            print("Model not trained. Please train the model before extracting coefficients.")
            return None

    def run(self):
        self.preprocess_data()
        self.split_data()
        self.train_and_evaluate_model()
        self.extract_prediction_probabilities()
        self.extract_coefficients_and_odds_ratios()
        self.apply_model_to_processed_data()
