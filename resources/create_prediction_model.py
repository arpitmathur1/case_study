import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# Import data
df = pd.read_csv('../data_clean/SKU_101.csv')

futureValue = df['Sales'].values.tolist()[1:]
futureValue.append(0)
df.insert(3, "future Sales", futureValue, True)
del(futureValue)


df.drop(['SKU', 'ISO_Week'], axis=1, inplace=True)
print(df.head())
threshold = int(len(df)*0.85)
train = df[:threshold]
test = df[threshold:]
del(df)
del(threshold)

trainY = train['future Sales']
testY = test['future Sales']

trainX = train.drop(['future Sales'], axis=1)
testX = test.drop(['future Sales'], axis=1)
del(train)
del(test)

print(trainX.head())
print(trainY.head())

regressor = SVR(kernel='rbf', verbose=True, gamma='auto')
regressor.fit(trainX, trainY)
# print(regressor)

predictions = regressor.predict(testX)
# print(predictions)
# print(testY)

plt.plot(predictions)
plt.plot(testY)
plt.show()



