# #####################################3
# Name             : clean_data.py    #3
# Author           : Arpit Mathur     #3
# #####################################3

import pandas as pd

file_path = "../data_orig/case_study_ML.xlsx"
csv_file_path = "../data_orig/case_study_ML.csv"

case_study_data = pd.read_excel(file_path)
case_study_data.to_csv(csv_file_path, index=False)

print('basic information about the database')
case_study_data.info()
print('data head')
print(case_study_data.head())


def plotting(dat):
    import matplotlib.pyplot as plt
    plt.plot(dat['ISO_Week'], dat['Sales'])
    plt.title("Item {0}".format(dat['SKU'].iloc[0]))
    plt.ylabel('Sales')
    plt.xlabel('Date')
    plt.show()


print('\n\ndescriptive statistics\n')
uniqueColNames = case_study_data.SKU.unique()
for colName in uniqueColNames:
    print('\n\n\t\t Item : {0}'.format(colName))
    uniqueData = case_study_data[case_study_data['SKU'] == colName]
    plotting(uniqueData)
    print("Column 'Sales' Mean {0}\nSTD : {1}".format(
            uniqueData['Sales'].mean(), uniqueData['Sales'].std()))







