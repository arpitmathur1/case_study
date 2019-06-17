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

print('dataset - promotion')
promotion = pd.read_csv('../data_orig/promotion.csv')
promotion.rename(columns={
        'FU': 'SKU',
        'Weeks': 'ISO_Week'
        }, inplace=True)
promotion['Promotion'] = 1
print(promotion.head())
print(promotion.describe())


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
    # obtain unique rows for that particular consumer good
    uniqueData = case_study_data[case_study_data['SKU'] == colName]
    # filling all NaN's with mean values
    uniqueData.fillna(uniqueData.mean(), inplace=True)

    # plotting to get better idea
    plotting(uniqueData)
    # obtaining mean and Standard Deviation of Sales data
    print("Column 'Sales' Mean {0}\nSTD : {1}".format(
            uniqueData['Sales'].mean(), uniqueData['Sales'].std()))

    # merging dataframe with promotion dataframe as a left join
    # and then replace all NaN's with default value 0, denoting not a promotion
    mergedDataframe = pd.merge(uniqueData, promotion, on=['SKU', 'ISO_Week'],
                               how='left')
    mergedDataframe.fillna(0, inplace=True)

    # showcase the data frame
    print(mergedDataframe)
