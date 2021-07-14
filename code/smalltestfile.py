datafilepath = '/home/nfs/mavantwout/data/CocaCola/missing90/KO_Shifted_30ahead.csv'
columnname = 'Close_ahead30_missing90_cubicfit30'
#I will adapt this to work with the specified column name in args instead of adapting the file name first. 

def getDataInfo(datafilename):
    split_on_data = datafilename.partition('/data/')
    split_for_missing =  split_on_data[2].partition('/missing')
    company = split_for_missing[0]
    missingness = split_for_missing[2][:2]

    split_for_imputation= columnname.split('_')
    imputation = split_for_imputation[-1]

    print(company)
    print(missingness)
    print(imputation)

getDataInfo(datafilepath)