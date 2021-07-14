import matplotlib.pyplot as plt
import seaborn as sns
tips = sns.load_dataset("tips")

tips_man = tips[tips['sex']=='Male']

tips_man = tips_man.groupby(['day','smoker']).total_bill.mean()

tips_man = tips_man.reset_index()
print("end")

