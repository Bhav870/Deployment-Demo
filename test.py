import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

model =joblib.load("model.pkl")

test_df=pd.read_csv("Iris.csv")
df=test_df.drop("Id", axis=1)
x=df.drop(['Species'], axis=1)
y=df['Species']

pred=model.predict(x)
accuracy = accuracy_score(pred,y)
print("Accuracy Score:", accuracy)

assert accuracy>=0.81
assert len(pred) == len(x) 

print('Test cases passed')