datafilename = '/home/nfs/mavantwout/data/CocaCola/missing90/KO-m90-1imputation-forwardfill.csv'

def getDataInfo(datafilename):
    split_on_data = datafilename.partition('/data/')
    split_for_missing =  split_on_data[2].partition('/missing')
    company = split_for_missing[0]
    missingness = split_for_missing[2][:2]
    split_for_imputation= datafilename.partition('imputation-')
    imputation = split_for_imputation[2].partition('.')[0]

    print(company)
    print(missingness)
    print(imputation)

getDataInfo(datafilename)