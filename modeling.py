"""This script models the European Soccer Data"""

import pandas as pd
import sqlite3 as sql
import pickle


class DataReader:
    def save_to_disk(self, filename):
        with open(filename, 'ab') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_disk(filename):
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        return obj


class ReadSQLDatabase(DataReader):
    """Read sql data bases into pandas dataframe: Each pandas object read from sql 
    gets set as an attribute named with the table name"""

    def __init__(self, sql_database, *tab_names):
        for tab in tab_names:
            setattr(self, tab, self.read_sql_table(tab, sql.connect(sql_database)))      
    
    def read_sql_table(tab_name, conn):
        pandas_dataframe = pd.read_sql(
            f"""
            select *
            from {tab_name}
            """, 
            conn)
        return pandas_dataframe


class ReadExcelDatabase(DataReader):
    def __init__(self, pandas_excel_reader, file_obj, **kwargs):
        setattr(self, 'data', pandas_excel_reader(file_obj, **kwargs)) 


class EuropeanSoccerDatabase(ReadSQLDatabase):
    pass


class NetflixFilms(ReadExcelDatabase):
    pass
