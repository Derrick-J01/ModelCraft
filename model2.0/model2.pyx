from xgboost import XGBRegressor 
import pandas as pd 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 



test_path = "test.csv" 
train_path = "train.csv" 

test = pd.read_csv(test_path) 
train = pd.read_csv(train_path) 

pd.set_option("display.max_columns", None) 
pd.set_option("display.max_rows", None)
pd.set_option("display.width", None)



train.plot.scatter(x="Foundation", y="SalePrice")

plt.show() 


# features = ["GrLivArea", "YearBuilt" ] 
# X =  

# y = data.SalesPrice  


# x_train, x_val, y_train, y_val = train_test_split(X, y, random_state=0) 

# model = XGBRegressor(n_estimators=1000) 






