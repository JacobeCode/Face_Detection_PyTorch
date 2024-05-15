import pyodbc as pyd
import pandas as pd

class connector():
    def __init__(self):
        self.driver = '{ODBC Driver 18 for SQL Server}'
        self.server = r'JACOBE_DESKTOP\EXPERIMENT_DATA'
        self.db = 'expression'
        self.uid = ''
        self.passwd = ''

    def query(self, query):
        # Connecting and executing query
        connection = pyd.connect(driver = self.driver, host = self.server, database = self.db, trusted_connection = 'yes')
        cursor = connection.cursor()

        cursor.execute(query)
        data=cursor.fetchall()

        data = pd.read_sql(query, connection)

        connection.close()

        return data