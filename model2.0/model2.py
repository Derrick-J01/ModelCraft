from xgboost import XGBRegressor 
import pandas as pd 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder 
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline
import pandas as pd 
from xgboost import XGBRegressor 	
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_log_error 

test_path = "test.csv" 
train_path = "train.csv" 

test = pd.read_csv(test_path) 
train = pd.read_csv(train_path) 

#pd.set_option("display.max_columns", None) 
#pd.set_option("display.max_rows", None)
#pd.set_option("display.width", None)

#train.plot.scatter(x="CentralAir", y="SalePrice")

#plt.show() 

features = ["GrLivArea", "YearBuilt", "CentralAir", "Neighborhood", "MSZoning"]

X = train[features] 

y = train.SalePrice


x_train, x_val, y_train, y_val = train_test_split(X, y, random_state=0)

ordinal_cols = ["CentralAir"] 
onehot_cols = ["Neighborhood","MSZoning"] 

ordinal_categories = [["N", "Y"]] 

numerical_cols = X.select_dtypes(include="number").columns.tolist()    


preprocessor = ColumnTransformer(transformers=[ 
 (
	"numerical", 
	SimpleImputer(strategy="median"), 
	numerical_cols
),    	
	

 (
	"ordinal",
	 Pipeline([
	("impute", SimpleImputer(strategy="most_frequent")),
	("encoder", OrdinalEncoder(
		categories=ordinal_categories,
		handle_unknown="use_encoded_value",
		unknown_value=-1
	))
	]), 
	ordinal_cols
	), 
	  

(
	"onehot", 
	Pipeline([ 
	("imputer", SimpleImputer(strategy="most_frequent")), 
	("encoder", OneHotEncoder(handle_unknown="ignore"))
	]), 
	onehot_cols
	) 	

]) 



X_train_processed = preprocessor.fit_transform(x_train)
X_val_processed =  preprocessor.transform(x_val) 




model = XGBRegressor(
	n_estimators=1000,
	 max_depth=4, 
	learning_rate=0.05, 
	early_stopping_rounds=4,
	random_state=0 
	)

model.fit(
	X_train_processed, 
	y_train, 
	eval_set=[(X_val_processed, y_val)], 
	verbose=False
)  


predictions = model.predict(X_val_processed)  

MAE = mean_absolute_error(y_val, predictions) 
print("MAE:", MAE) 

RMSLE = mean_squared_log_error( 
	y_val, 
	predictions.clip(min=0) 
	) ** 0.5 

print("RMSLE:", RMSLE) 
