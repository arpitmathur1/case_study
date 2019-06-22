"""
TODO's -- 22-06-2019 1:38 AM

1) make files relative to file - DONE
2) make the models hyper-parameter tuned - WIP#1
3) obtain the best model and push the predicted values to a separate DataFrame
   and push the DataFrame to an excel sheet / CSV file - DONE
"""


import pandas as pd
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import os
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
# matplotlib warnings on my system
# ## UserWarning: Matplotlib is currently using agg, which is a non-GUI
# ## backend, so cannot show the figure.
warnings.filterwarnings("ignore")

# matplotlib config setting
matplotlib.use('Agg')


# setting relative paths, W.R.T. this file
filePath = os.path.realpath(__file__)
filePath = filePath[:filePath.rfind('\\')]

# input files which have cleaned data with respect to their datasets
files = [filePath + '\\..\\data_clean\\SKU_101.csv',
         filePath + '\\..\\data_clean\\SKU_102.csv',
         filePath + '\\..\\data_clean\\SKU_103.csv'
         ]

modelParameterSavingFiles = [
        filePath + '\\..\\model_history\\SupportVector.csv',
        filePath + '\\..\\model_history\\RandomForest.csv',
        filePath + '\\..\\model_history\\LinearRegression.csv',
        filePath + '\\..\\model_history\\NNRegressor.csv'
        ]

final = pd.DataFrame()

# finding the optimal model in case of each sub-data-set
for file in files:

    OptimalModel = None
    OptimalMetricMAE = 100000
    OptimalMetrics = {}

    print("\n\ninput file: {0}".format(file))
    itemName = file[file.rfind('\\')+1:file.rfind('.')]
    print('........>>>>>>' + itemName)

    modelHistoryFile = open(filePath+"\\..\\model_history\\{0}_modelHistory.csv".format(
                itemName), 'a')
    now = datetime.now()
    modelHistoryFile.write("Run Start Time: {0}\n\n".format(now.strftime("%m/%d/%Y, %H:%M:%S")))
    modelHistoryFile.write('model type,kernel type,MAE value,MSE Value')
    modelHistoryFile.write('\n')
    modelHistoryFile.flush()
    modelHistoryFile.close()

    # Import data
    df = pd.read_csv(file)

    # ideate the threshold - the place where the split shall happen
    # acc. to the word doc, it has to be at 46th row (test rows start)
    threshold = int(len(df)*0.85)

    # dropping rows where sales value is zer (acc. to documentation provided)
    # df = df[df.Sales != 0]

    # generate the 'future values' column
    futureValue = df['Sales'].values.tolist()[1:]
    futureValue.append(0)
    df.insert(3, "future Sales", futureValue, True)
    del(futureValue)

    # dropping unnecessary columns
    df.drop(['SKU', 'ISO_Week'], axis=1, inplace=True)

    # creating train-test split
    train = df[:threshold]
    test = df[threshold:]
    names = df.columns.values
    del(df)
    del(threshold)

    # generating the X, Y dataframe
    trainY = train['future Sales']
    testY = test['future Sales']
    trainX = train.drop(['future Sales'], axis=1)
    testX = test.drop(['future Sales'], axis=1)
    del(train)
    del(test)

    # ########## Generating SVR Model ##########

    kernel = ['linear', 'poly', 'rbf', 'sigmoid']

    for singleKernel in kernel:
        regressor = SVR(kernel=singleKernel, verbose=False)
        regressor.fit(trainX, trainY)

        filename = filePath + '\\..\\models\\SVR\\{0}_initial_SVR_{1}_model.savefile'.format(
            file[file.rfind('\\'):file.rfind('.')], singleKernel)
        pickle.dump(regressor, open(filename, 'wb'))
        predictions = regressor.predict(testX)

        MSE = mean_squared_error(predictions, testY)
        MAE = mean_absolute_error(predictions, testY)

        plt.plot(predictions)
        plt.plot(testY)
        plt.xlabel('compare predictions')
        plt.ylabel('sales value (scaled)')
        plt.title('SVR Kernel - {0} - data for {1}'.format(
                singleKernel, file[file.rfind('\\')+1:file.rfind('.')]))
        plt.show()
        plt.savefig(filePath + '\\..\\visualizations\\SVR\\{0}_SVR_{1}.jpg'.format(
                file[file.rfind('\\'):file.rfind('.')],
                singleKernel))
        plt.close()

        modelHistoryFile = open(filePath+"\\..\\model_history\\{0}_modelHistory.csv".format(
                itemName, singleKernel), 'a')
        modelHistoryFile.write('SVR,{0},{1},{2}'.format(
                singleKernel,
                MAE,
                MSE
                )
        )
        modelHistoryFile.write('\n')
        modelHistoryFile.flush()
        modelHistoryFile.close()

        if OptimalMetricMAE > MAE:
            OptimalMetricMAE = MAE
            OptimalModel = regressor
            OptimalMetrics = {
                    'MAE': MAE,
                    'MSE': MSE,
                    'kernalName': singleKernel,
                    'model_type': 'SVR'
                    }
        else:
            del(regressor)

    # #################3 Generate Random Forest Regressor Model ###
    estimators = range(10,510,10)
    estimators = range(10,30,10)
    criterias = ['mae', 'mse']
    depths = [None, 2,4,8,16,32,64]
    depths = [None, 2,4,8]

    for estimatorCount in estimators:
        for criteria in criterias:
            for depth in depths:
                regressor = RandomForestRegressor(criterion=criteria,
                                                  n_estimators=estimatorCount,
                                                  n_jobs=-1,
                                                  max_depth=depth
                                                  )
                regressor.fit(trainX, trainY)

                filename = filePath + '\\..\\models\\RFR\\{0}_RFR_{1}_{2}_{3}_model.savefile'.format(
                        file[file.rfind('/'):file.rfind('.')],
                        estimatorCount,
                        criteria,
                        depth
                        )
                pickle.dump(regressor, open(filename, 'wb'))

                predictions = regressor.predict(testX)

                MSE = mean_squared_error(predictions, testY)
                MAE = mean_absolute_error(predictions, testY)

                plt.plot(predictions)
                plt.plot(testY)
                plt.xlabel('compare predictions')
                plt.ylabel('sales value (scaled)')
                plt.title('RFR - estimators:{0} - criteria:{1} - max_depth:{2}'.format(
                        estimatorCount,
                        criteria,
                        depth
                        )
                )
                plt.show()
                plt.savefig(filePath + '\\..\\visualizations\\RFR\\{0}_{1}_{2}_{3}_RFR.jpg'.format(
                        file[file.rfind('\\'):file.rfind('.')],
                        estimatorCount,
                        criteria,
                        depth
                        )
                )
                plt.close()

                modelHistoryFile = open(filePath+"\\..\\model_history\\{0}_modelHistory.csv".format(
                        itemName), 'a')
                modelHistoryFile.write('RFR,{0}-{1}-{2},{3},{4}'.format(
                        estimatorCount,
                        criteria,
                        depth,
                        MAE,
                        MSE
                        )
                )
                modelHistoryFile.write('\n')
                modelHistoryFile.flush()
                modelHistoryFile.close()

                if OptimalMetricMAE > MAE:
                    OptimalMetricMAE = MAE
                    OptimalModel = regressor
                    OptimalMetrics = {
                            'MAE': MAE,
                            'MSE': MSE,
                            'estimator_count': estimatorCount,
                            'criteria': criteria,
                            'max_depth': depth,
                            'model_type': 'RFR'
                            }
                else:
                    del(regressor)

    # #################3 Generate Linear Regression Model ###

    regressor = LinearRegression(n_jobs=-1)
    regressor.fit(trainX, trainY)

    filename = filePath + '\\..\\models\\linear\\{0}_initial_Linear_model.savefile'.format(
            file[file.rfind('\\'):file.rfind('.')])
    pickle.dump(regressor, open(filename, 'wb'))
    predictions = regressor.predict(testX)

    MSE = mean_squared_error(predictions, testY)
    MAE = mean_absolute_error(predictions, testY)

    plt.plot(predictions)
    plt.plot(testY)
    plt.xlabel('compare predictions')
    plt.ylabel('sales value (scaled)')
    plt.title('Linear Regression')
    plt.show()
    plt.savefig(filePath + '\\..\\visualizations\\linear\\{0}_Linear.jpg'.format(
            file[file.rfind('\\'):file.rfind('.')]))
    plt.close()

    modelHistoryFile = open(filePath+"\\..\\model_history\\{0}_modelHistory.csv".format(
        itemName), 'a')
    modelHistoryFile.write('LR,LR,{0},{1}'.format(
        MAE,
        MSE
        )
    )
    modelHistoryFile.write('\n')
    modelHistoryFile.flush()
    modelHistoryFile.close()

    if OptimalMetricMAE > MAE:
        OptimalMetricMAE = MAE
        OptimalModel = regressor
        OptimalMetrics = {
                'MAE': MAE,
                'MSE': MSE,
                'model_type': 'LR'
                }
    else:
        del(regressor)

    predY = OptimalModel.predict(testX)
    finalDataFrame = testX
    print('........>>>>>>' + itemName)
    finalDataFrame.insert(0, 'SKU', itemName)
    finalDataFrame.insert(4, 'Forecast', predY)
    print(finalDataFrame)
    finalDataFrame.reset_index(drop=True)
    final = final.append(finalDataFrame)
    print(OptimalMetrics)
final.to_csv(filePath + '\\..\\final_data\\expected_output.csv', index=False)
