import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("creditcard.csv")
# data.info()
y = data['Class']
x = data.drop('Class', axis=1)
x = x.drop('Time', axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4)

rf=RandomForestClassifier()
rf.fit(x_train,y_train)

print(help(RandomForestClassifier))
out6=rf.predict(x_test)

print("Random Forest training score:", rf.score(x_train,y_train))
print("Random Forest testing score:", accuracy_score(y_test,out6))
