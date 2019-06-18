# EY_case_study

## Steps taken till now - Data Cleaning [18-06-2019 1:18AM]
0. created a csv file with details pertaining to promotion weeks and their products
1. Saved the 'case_study_ML' excel sheet as a csv -- python line 8-12, 
2. obtained the basic description of the dataframe 'case_study_ML' as well as 'promotion' -- python line 14-17
3. obtained the 'promotion' dataset from the csv file, after re-naming columns to original data frame standard (shall be used later on), as well as adding a new column, called 'promotion' set with default integer value '1' -- python line 19-27
4. created a plotting method to observe the individual product-id's data -- python line 30-36
5. obtained a list of unique product ID's from the dataframe and used it to -

    5.1. replace all NaN values with the mean one for the dataframe

    5.2. plot each individual's data

    5.3. Merge the 'case_study_ML' dataframe with the 'promotion' one with a left join

    5.4. After merging, replace all 'NaN' values in the 'promotion' column with the integer '0'

    5.5. Replace the values in the 'Season' column with 0-3 according to label encoding method

    5.6. save the individual product's data to a file in the cleaned data location

6. copy the individual product's data to a python list, and use that list to concatenate each other with index false (to re-create the index) and save it as a separate cleaned data file for future use


## External Data being called
Link: https://joshuaproject.net/resources/datasets

Steps:
1. download the database
2. export the table 'tblGEO3Countries' to excel sheet
