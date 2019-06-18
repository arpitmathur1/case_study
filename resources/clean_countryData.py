import pyodbc
import pandas as pd


conn = pyodbc.connect(r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=H:\project\EY\EY_case_study\data_external\JPHarvestField.accdb;')
cursor = conn.cursor()
cursor.execute("""select * from tblGEO3Countries""")

data = []
for row in cursor.fetchall():
    data.append(list(row))

df = pd.DataFrame(data)
print(df.head())

df.to_csv('../data_external/countryData.csv', index=False)
