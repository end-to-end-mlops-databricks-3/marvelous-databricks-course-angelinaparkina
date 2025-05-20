""" Data Reader module.""" 


import pandas as pd

class DataReader:
    """Takes care of reading in the data. It needs a path pointing to where the data resides.

    Right now only works for csv format.
    It will read in and return a pandas dataframe. 
    
    """
    def __init__(self,file_path: str) -> None:
        #do i also need to specify requirements for file path into the parameters..or should i place the path into project config and read it in from there
        self.path = file_path

    def read_csv(self) -> pd.DataFrame:
        #potentially this needs to be more flexible with format specified as parameter

        df = pd.read_csv(self.path,header=0)

        return df


