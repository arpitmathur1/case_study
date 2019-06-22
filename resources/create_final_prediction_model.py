"""
TODO's -- 22-06-2019 1:38 AM

1) make files relative to file - DONE
2) make the models hyper-parameter tuned
3) obtain the best model and push the predicted values to a separate DataFrame
   and push the DataFrame to an excel sheet / CSV file
"""


import pandas as pd
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import os
import pickle
from scipy.stats.stats import pearsonr
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
# matplotlib config setting
matplotlib.use('Agg')


# setting relative paths, W.R.T. this file
filePath = os.path.realpath(__file__)
print(filePath)
filePath = filePath[:filePath.rfind('\\')]
print(filePath)

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

OptimalModel = None
OptimalMetricMAE = 100000
OptimalMetrics = {}

# finding the optimal model in case of each sub-data-set
for file in files:
    print(file)
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
    df = df[df.Sales != 0]

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

        filename = filePath + '\\..\\models\\{0}_initial_SVR_{1}_model.savefile'.format(
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
                singleKernel,file[file.rfind('\\')+1:file.rfind('.')]))
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

        del(regressor)
    """
    # #################3 Generate Random Forest Regressor Model ###

    regressor = RandomForestRegressor(criterion="mae",
                                      n_estimators=100,
                                      n_jobs=-1,
                                      max_depth=6,
                                      verbose=1
                                      )
    regressor.fit(trainX, trainY)
    print('Random Forest regression score')
    print(regressor.score(trainX, trainY))

    filename = filePath + '\\..\\models\\{0}initial_RFR_model.savefile'.format(
            file[file.rfind('/'):file.rfind('.')])
    pickle.dump(regressor, open(filename, 'wb'))

    predictions = regressor.predict(testX)
    print(predictions)
    print(testY)

    pearson_correlationValues = pearsonr(predictions, testY)
    print("\ncorrelation = " + str(pearson_correlationValues[0]))
    print("significance = " + str(pearson_correlationValues[1]))
    MSE = mean_squared_error(predictions, testY)
    MAE = mean_absolute_error(predictions, testY)
    print("MSE = {0} \nMAE = {1}".format(MSE, MAE))

    plt.plot(predictions)
    plt.plot(testY)
    plt.xlabel('compare predictions')
    plt.ylabel('sales value (scaled)')
    plt.title('RFR trial One')
    plt.show()
    plt.savefig(filePath + '\\..\\visualizations\\{0}RFR_trialOne.jpg'.format(
            file[file.rfind('\\'):file.rfind('.')]))
    plt.close()

    del(regressor)
    # #################3 Generate Linear Regression Model ###

    regressor = LinearRegression(n_jobs=-1)
    regressor.fit(trainX, trainY)
    print('linear regression score')
    print(regressor.score(trainX, trainY))

    filename = filePath + '\\..\\models\\{0}_initial_Linear_model.savefile'.format(
            file[file.rfind('\\'):file.rfind('.')])
    pickle.dump(regressor, open(filename, 'wb'))

    predictions = regressor.predict(testX)

    pearson_correlationValues = pearsonr(predictions, testY)
    print("\ncorrelation = " + str(pearson_correlationValues[0]))
    print("significance = " + str(pearson_correlationValues[1]))
    MSE = mean_squared_error(predictions, testY)
    MAE = mean_absolute_error(predictions, testY)
    print("MSE = {0} \nMAE = {1}".format(MSE, MAE))

    plt.plot(predictions)
    plt.plot(testY)
    plt.xlabel('compare predictions')
    plt.ylabel('sales value (scaled)')
    plt.title('Linear Regression trial One')
    plt.show()
    plt.savefig(filePath + '\\..\\visualizations\\{0}_Linear_trialOne.jpg'.format(
            file[file.rfind('\\'):file.rfind('.')]))
    plt.close()

    del(regressor)
    # #################3 NN regression model generation ###

    regressor = MLPRegressor(hidden_layer_sizes=(100, ),
                             activation='relu',
                             solver='sgd',
                             learning_rate='adaptive',
                             learning_rate_init=0.1,
                             shuffle=False
                             )

    regressor.fit(trainX, trainY)
    print('NN score')
    print(regressor.score(trainX, trainY))

    filename = filePath + '\\..\\models\\{0}_initial_Linear_model.savefile'.format(
            file[file.rfind('\\'):file.rfind('.')])
    pickle.dump(regressor, open(filename, 'wb'))

    predictions = regressor.predict(testX)

    pearson_correlationValues = pearsonr(predictions, testY)
    print("\ncorrelation = " + str(pearson_correlationValues[0]))
    print("significance = " + str(pearson_correlationValues[1]))
    MSE = mean_squared_error(predictions, testY)
    MAE = mean_absolute_error(predictions, testY)
    print("MSE = {0} \nMAE = {1}".format(MSE, MAE))

    plt.plot(predictions)
    plt.plot(testY)
    plt.xlabel('compare predictions')
    plt.ylabel('sales value (scaled)')
    plt.title('NN Regression trial One')
    plt.show()
    plt.savefig(filePath + '\\..\\visualizations\\{0}_NN_trialOne.jpg'.format(
            file[file.rfind('\\'):file.rfind('.')]))
    plt.close()
    del(regressor)
    """
