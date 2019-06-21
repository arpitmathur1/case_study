# AUTHOR : Arpit Mathur
# Script Creation date: 22-06-2019 1:35AM
# execute everything

try:
    # this shall execute the entirity of the cleaning data script
    # since it doesn't contain any class / methods
    import resources.clean_data
except Exception as ex:
    print(ex)

try:
    # this shall execute the entirity of the creation of the model script
    import resources.create_final_prediction_model
except Exception as ex:
    print(ex)
