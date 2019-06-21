import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import pickle
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Import data
df = pd.read_csv('../data_clean/SKU_101.csv')

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
regressor = SVR(kernel='rbf', verbose=True, gamma='auto')
regressor.fit(trainX, trainY)
print('SVR regression score')
print(regressor.score(trainX, trainY))

filename = '../models/initial_SVR_model.savefile'
pickle.dump(regressor, open(filename, 'wb'))

predictions = regressor.predict(testX)
# print(predictions)
# print(testY)

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
plt.title('SVR trial One')
plt.show()
plt.savefig('../visualizations/SVR_trialOne.jpg')
plt.close()

del(regressor)
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

filename = '../models/initial_RFR_model.savefile'
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
plt.savefig('../visualizations/RFR_trialOne.jpg')
plt.close()

del(regressor)
# #################3 Generate Linear Regression Model ###

regressor = LinearRegression(n_jobs=-1)
regressor.fit(trainX, trainY)
print('linear regression score')
print(regressor.score(trainX, trainY))

filename = '../models/initial_Linear_model.savefile'
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
plt.title('Linear Regression trial One')
plt.show()
plt.savefig('../visualizations/Linear_trialOne.jpg')
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

filename = '../models/initial_Linear_model.savefile'
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
plt.title('NN Regression trial One')
plt.show()
plt.savefig('../visualizations/NN_trialOne.jpg')
plt.close()
del(regressor)
