datafilepath = './thesis_datasets/Dataset4/AAPL_Price_and_Quarterly.csv'
columnname = 'Close_ahead30_missing90_cubicfit30'
#I will adapt this to work with the specified column name in args instead of adapting the file name first. 

# def getDataInfo(datafilename, columnname):
#     split_on_data = datafilename.partition('/data/')
#     split_for_missing =  split_on_data[2].partition('/missing')
#     company = split_for_missing[0]
#     missingness = split_for_missing[2][:2]

#     split_for_imputation= columnname.split('_')
#     imputation = split_for_imputation[-1]

#     print(company)
#     print(missingness)
#     print(imputation)

# getDataInfo(datafilepath, columnname)
#Experiment 5
def getDataInfo(datafilename, columnname):
    split_on_data = datafilename.partition('/Dataset4/')
    split_for_missing =  split_on_data[2].partition('_Price')
    company = split_for_missing[0]

    split_for_imputation= columnname.split('_')
    imputation = split_for_imputation[-1]

    print(company)
    print(imputation)

getDataInfo(datafilepath, columnname)
