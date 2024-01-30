import pandas as pd

class DataHandler:
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

    def export_data(self, data, export_path):
        """
        Exports the provided DataFrame to the specified path.
        
        Parameters:
        - data (DataFrame): The pandas DataFrame to export.
        - export_path (str): The file path where the DataFrame will be saved.
        
        Returns:
        - None
        """
        if not isinstance(data, pd.DataFrame):
            print("Error: The data provided is not a pandas DataFrame.")
            return

        if not export_path:
            print("Error: No export path provided.")
            return

        try:
            data.to_csv(export_path, index=False)
            print(f"Data exported successfully to {export_path}.")
        except Exception as e:
            print(f"An error occurred while exporting the data: {e}")