""" Data Reader module."""

import pandas as pd

class DataReader:
    """Takes care of reading in the data. It needs a path pointing to where the data resides.

    Right now only works for csv format.
    It will read in and return a pandas dataframe.
    
    """
    
    def __init__(self,file_path: str) -> None:
        self.path = file_path

    def read_csv(self) -> pd.DataFrame:
        """Read in csv file and returns a pandas dataframe."""
        
        df = pd.read_csv(self.path,header=0) #potentially this needs to be more flexible with format specified as parameter
        
        return df
