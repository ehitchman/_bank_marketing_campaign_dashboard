import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer

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

    def _finalize_variables(self):
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

        print('\n----------------------------')
        print('PREPROCESS DATA'.center(28, '-'))
        print('----------------------------')

        print('\n----------------------------')
        print('raw data types'.center(28, '-'))
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
        self.processed_data = pd.get_dummies(self.processed_data, columns=[new_attr], drop_first=False)
        
        # Convert age columns to int
        age_columns = [col for col in self.processed_data.columns if 'age_' in col]
        self.processed_data[age_columns] = self.processed_data[age_columns].astype(int)
 
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
        self.processed_data.drop(attr, axis=1, inplace=True)

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
        self._finalize_variables()

        print('\n----------------------------')
        print('model input data preview/types'.center(28, '-'))
        print(self.processed_data.head(5))
        print(self.processed_data.dtypes)
    
    def split_data(self):
        """
        Split data into training and testing sets
        """
        x = self.processed_data.drop(self.target_variable, axis=1)
        y = self.processed_data[self.target_variable]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=self.test_size, random_state=self.random_state)
        
        print('\n----------------------------')
        print('SPLIT TRAINING/TESTING'.center(28, '-'))
        print('----------------------------')

        print('----------------------------')
        print('self.x_train'.center(28, '-'))
        print(self.x_train.head(5))

        print('\n----------------------------')
        print('self.x_test'.center(28, '-'))
        print(self.x_test.head(5))

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
        self.model.fit(self.x_train, self.y_train)
        self.y_pred = self.model.predict(self.x_test)
        print('----------------------------')
        
        print('\n----------------------------')
        print('TRAIN AND EVALUATE MODEL'.center(28, '-'))

        print('\n----------------------------')
        print('classification report'.center(28, '-'))
        print(classification_report(self.y_test, self.y_pred))
        
        print('\n----------------------------')
        print('confusion matrix'.center(28, '-'))
        print(confusion_matrix(self.y_test, self.y_pred))

    def extract_prediction_probabilities(self):
        """
        Extracts the prediction probabilities for the test data set and attaches 
        them to the data set.
        """
        print('\n----------------------------')
        print('EXTRACT PREDICTIONS'.center(28, '-'))
        print('----------------------------')
        
        self.predicted_probabilities = self.model.predict_proba(self.x_test)
        print('\n----------------------------')
        print('predicted probabilities'.center(28, '-'))
        print(self.predicted_probabilities[:5])

        
        self.x_test['predicted_probabilities'] = self.predicted_probabilities[:, 1] 
        print('\n----------------------------')
        print('attached probabilities'.center(28, '-'))
        print(self.x_test.head(5))

        return self.x_test

    def run(self):
        self.preprocess_data()
        self.split_data()
        self.train_and_evaluate_model()
        prediction_probabilities = self.extract_prediction_probabilities()
        return prediction_probabilities
