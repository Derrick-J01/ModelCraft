import pandas as pd 
from sklearn.ensemble import RandomForestRegressor  
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split 
from sklearn.impute import SimpleImputer  
from sklearn.metrics import mean_squared_log_error
import math 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score 
import numpy as np 


pd.set_option("display.max_columns", None) 
pd.set_option("display.max_rows", None) 
pd.set_option("display.width", None) 


test="test.csv"
train="train.csv"

trainhouses=pd.read_csv(train)

testhouses=pd.read_csv(test)  

print(trainhouses.head(5)) 

print(trainhouses.describe())


trainhouses=trainhouses.dropna(subset=["SalePrice"])

features=["LotArea","YearBuilt","PoolArea","Neighborhood","OverallCond", "OverallQual", "BsmtQual","BsmtCond","MSZoning"] 

X=trainhouses[features] 
y=trainhouses.SalePrice 
numerical_cols = X.select_dtypes(include="number").columns.tolist()   


onehot_cols = ["Neighborhood", "MSZoning"]  
ordinal_cols= ["BsmtQual","BsmtCond"] 

ordinal_categories=[
["NA","Po","Fa","TA","Gd","Ex"],
["NA","Po","Fa","TA","Gd","Ex"]
]  

preprocessor = ColumnTransformer(
	transformers=[
	("numerical", SimpleImputer(strategy="median"), numerical_cols),

	("onehot",
	
	 Pipeline([
	("imputer", SimpleImputer(strategy="most_frequent")),
	("encoder", OneHotEncoder(handle_unknown="ignore"))
	]),
 
	onehot_cols), 
	
	("ordinal",
	
	Pipeline([ 
	("imputer", SimpleImputer(strategy="most_frequent")),
	("encoder", OrdinalEncoder(categories=ordinal_categories,handle_unknown="use_encoded_value", unknown_value=-1))
 	]), 
	ordinal_cols)
	]
)

model = RandomForestRegressor(random_state=0, n_estimators=100)


my_pipeline = Pipeline(steps=[
	("preprocessor", preprocessor),
	("model", model)
	]
) 

scores = -1 * cross_val_score( 
	my_pipeline, 
	X,
	y, 
	cv=5, 
	scoring="neg_mean_absolute_error" 
) 

print("MAE scores:", scores)
print("Average MAE:", scores.mean())
print("Minimum MAE:", scores.min())
print("Maximum MAE:", scores.max())
print("Standard deviation:", scores.std())

X_encoded = preprocessor.fit_transform(X) 

model.fit(X_encoded, y) 

X_test_encoded = preprocessor.transform(testhouses[features]) 

test_prediction = model.predict(X_test_encoded) 
 

