import pandas as pd


df = pd.read_csv("bench_rdf2vec_correct_cbow.csv")
print(df.columns)
x = df[(df["Dataset"]=="AIFB") & (df["depth"]==4)].groupby(["Dataset","depth"]).mean()
print((x["Memory"]).values[0]/10**9)
x = df[(df["Dataset"]=="AIFB") & (df["depth"]==4)].groupby(["Dataset","depth"]).std()
print((x["Memory"]).values[0])

x = df[(df["Dataset"]=="AIFB") & (df["depth"]==4)].groupby(["Dataset","depth"]).mean()
print((x["Create_time"]+x["SVC_Train_time"]+x["SVC_Test_time"]).values[0])
x = df[(df["Dataset"]=="AIFB") & (df["depth"]==4)].groupby(["Dataset","depth"]).std()
print((x["Create_time"]+x["SVC_Train_time"]+x["SVC_Test_time"]).values[0])



#x = df[(df["Dataset"]=="AM") & (df["depth"]==1)].groupby(["Dataset","depth"]).mean()
#print((x["Create_time"]+x["LR_Train_time"]+x["LR_Test_time"]).values[0])
#x = df[(df["Dataset"]=="AM") & (df["depth"]==1)].groupby(["Dataset","depth"]).std()
#print((x["Create_time"]+x["LR_Train_time"]+x["LR_Test_time"]).values[0])

