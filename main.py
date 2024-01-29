#import data from .data/bank.csv

import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer


class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = self.load_data()

    def load_data(self):
        try:
            df = pd.read_csv(self.filepath)
            print("Data loaded successfully.")
            return df
        except FileNotFoundError:
            print("File not found. Please check the file path and try again.")
            return None
    
class DataModeler:
    def __init__(self, data, target_variable: str, numeric_vars: list, categorical_vars: list, test_size: float = 0.2, random_state: int = 42):
        
        # Initialize data
        self.data = data
        self.categorical_vars = categorical_vars
        self.numeric_vars = numeric_vars
        self.target_variable = target_variable

        # Initialize data preprocessing variables
        self.encoder = OneHotEncoder(sparse_output=False, drop='first')
        self.test_size = test_size
        self.random_state = random_state

        # Initialize model
        self.model = LogisticRegression(max_iter=1000)

        # Initialize data splits
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def finalize_variables(self):
        """
        Retains only the specified numeric variables, target variable, and encoded categorical variables in the processed data.
        """
        #TODO: Could potentially be automated using inclusion of 'continuous_vars' argument in function signature and
        #     checking if the variable is in the list of any variables (requires some review)
        # all_vars = self.numeric_vars + [self.target_variable] + list(self.encoder.get_feature_names_out(self.categorical_vars))
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
        print(self.processed_data.dtypes)

        # Encode poutcome using a binary variable based on whether the previous campaign was successful or not ('Previous
        # Campaign Outcome')
        attr = 'poutcome'
        new_attr = attr+'_binary'
        self.processed_data[new_attr] = self.processed_data[attr].apply(lambda x: 1 if x == 'success' else 0)
        self.processed_data = self.processed_data.drop(attr, axis=1)

        # Encode age with custom bucket sizes (note, we also use one-hot encoding here separately as its not a part of
        # the initial data set and we don't necessarily think that age is a linear relationship with the target variable)
        attr = 'age'
        new_attr = 'age_'+attr
        bins = [18, 30, 40, 50, 60, 150]
        bin_labels = ['18-29', '30-39', '40-49', '50-59', '60+']
        self.processed_data[new_attr] = pd.cut(self.processed_data[attr], bins=bins, labels=bin_labels, right=False)
        self.processed_data = self.processed_data.drop(attr, axis=1)
        self.processed_data = pd.get_dummies(self.processed_data, columns=[new_attr], drop_first=True)
 
        # Encode balance with binned values
        # TODO: Add as a conditional based on an argument 'continuous_vars: list = ['balance']' function signature
        # if self.continuous_vars:
        attr = 'balance'
        new_attr = 'binned_'+attr
        est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        self.processed_data[new_attr] = est.fit_transform(self.processed_data[[attr]])
        self.processed_data = self.processed_data.drop(attr, axis=1)

        # Encode campaign (in-campaign contacts)
        # TODO: Add as a conditional based on a argument 'capped_vars: list = ['balance']' function signature
        attr = 'campaign'
        new_attr = attr+'_capped'
        threshold = self.processed_data[attr].quantile(0.99)
        self.processed_data[new_attr] = self.processed_data[attr].apply(lambda x: min(x, threshold))
        self.processed_data.drop(attr, axis=1)

        # Encode previous (in-campaign contacts) using a binary variable based on whether they have been contacted or not
        # TODO: Add as a conditional based on a argument 'capped_vars: list = ['balance']' function signature
        attr = 'previous'
        new_attr = attr+'_binary'
        self.processed_data[new_attr] = self.processed_data[attr].apply(lambda x: 0 if x == -1 else 1)
        self.processed_data = self.processed_data.drop(attr, axis=1)

        # Encode pdays using a binary variable based on whether they had been contacted or not in the previous campaign
        # contacted from previous campaign ('Days Since Last Contacted')
        attr = 'pdays'
        new_attr = attr+'_yes'
        self.processed_data[new_attr] = self.processed_data[attr].apply(lambda x: 0 if x == -1 else 1)
        self.processed_data = self.processed_data.drop(attr, axis=1)

        # Encode all listed categorical variables using one-hot encoding
        if self.categorical_vars:
            encoded_vars = self.encoder.fit_transform(self.processed_data[self.categorical_vars])
            encoded_df = pd.DataFrame(encoded_vars, columns=self.encoder.get_feature_names_out(self.categorical_vars))
            print(encoded_df.head(5))
            self.processed_data = pd.concat([self.processed_data.drop(self.categorical_vars, axis=1), encoded_df], axis=1)

        # Encode target variable using one-hot encoding
        if self.processed_data[self.target_variable].dtype == 'object':
            self.processed_data[self.target_variable] = self.processed_data[self.target_variable].map({'yes': 1, 'no': 0})

        # Keep only the selected numeric variables, target, and encoded categorical variables
        self.finalize_variables()
    
    def split_data(self):
        """
        Split data into training and testing sets
        """
        x = self.processed_data.drop(self.target_variable, axis=1)
        y = self.processed_data[self.target_variable]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=self.test_size, random_state=self.random_state)
        
        print("\nself.x_train")
        print(self.x_train)
        print("\nself.x_test")
        print(self.x_test)
        print("\nself.y_train")
        print(self.y_train)
        print("\nself.y_test")
        print(self.y_test)

    def train_model(self):
        """
        Trains the logistic regression model using the training dataset.
        """
        self.model.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.x_test)

        print("This the classification report for the model.")
        print(classification_report(self.y_test, y_pred))
        
        print("This is the confusion matrix for the model.")
        print(confusion_matrix(self.y_test, y_pred))

    def run(self):
        self.preprocess_data()
        self.split_data()
        self.train_model()
        self.evaluate_model()

def main():
    
    # Create a data loader object and load data
    bank_csv_filepath = os.path.join('.', 'data', 'bank.csv')
    data_loader = DataLoader(filepath=bank_csv_filepath)
    data = data_loader.data

    # Create a data modeler object
    data_modeler = DataModeler(
        data=data, 
        target_variable='deposit', 
        numeric_vars=['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'],
        categorical_vars=['job', 'marital', 'education', 'default', 'housing', 'loan']
        )
    
    # Preprocess data
    data_modeler.preprocess_data()
    print(data_modeler.processed_data.head(5))
    print(data_modeler.processed_data.dtypes)

    # Split data
    print('\n----------------------------')
    print('SPLIT TRAINING/TESTING'.center(28, '-'))
    data_modeler.split_data()

    # Train model
    print('\n----------------------------')
    print('TRAIN MODEL'.center(28, '-'))
    data_modeler.train_model()

    # # Evaluate model
    print('\n----------------------------')
    print('EVALUATE MODEL'.center(28, '-'))
    data_modeler.evaluate_model()
    
def test_get_data():
    # Get the data
    bank_csv_filepath = os.path.join('.', 'data', 'bank.csv')
    data_loader = DataLoader(filepath=bank_csv_filepath)
    data = data_loader.get_data()
    
    print(data.head(5))
    return data

if __name__ == "__main__":
    data_loader = main()
