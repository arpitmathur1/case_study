# AUTHOR : Arpit Mathur
# Script Creation date: 22-06-2019 1:35AM
# execute everything

import os
import warnings

# matplotlib warnings on my system
# ## UserWarning: Matplotlib is currently using agg, which is a non-GUI
# ## backend, so cannot show the figure.
warnings.filterwarnings("ignore")

# get file paths, so that we can create the folders necessary
filePath = os.path.realpath(__file__)
filePath = filePath[:filePath.rfind('\\')]

# relative paths to be created
filePathsToCreate = [
        '\\data_clean',
        '\\final_data',
        '\\model_history',
        '\\models',
        '\\visualizations',
        '\\models\\linear',
        '\\models\\RFR',
        '\\models\\SVR',
        '\\visualizations\\linear',
        '\\visualizations\\RFR',
        '\\visualizations\\SVR'
        ]

print("creating folders")
# creating folders
for relFilePath in filePathsToCreate:
    try:
        print("creating> {0}".format(filePath + relFilePath))
        os.mkdir(filePath + relFilePath)
    except FileExistsError:
        print("folder already exists: {0}".format(filePath + relFilePath))
print("folders created")

print("data cleaning shall start")
try:
    # this shall execute the entirity of the cleaning data script
    # since it doesn't contain any class / methods
    import resources.clean_data
except Exception as ex:
    print(ex)
print("data cleaning completed")
print("model creation starts")
print("""\twe shall be using Linear Models, Random Forest Regressor Models,
      Support Vector Machine Models""")

try:
    # this shall execute the entirity of the creation of the model script
    import resources.create_final_prediction_model
except Exception as ex:
    print(ex)
print("model creation has ended")
print("===>")
print("please go to the following directories for additional information")
print(""" - visualization files per item per model  /visualizations
- visualization files per item after cleaning  /visualizations
- historic model files per item (CSV data)  /model_history
- historic model files per item /models
- all predictions based on optimized models /final_data""")
