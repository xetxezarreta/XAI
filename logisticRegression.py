import pandas
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

filename = "data/pima-indians-diabetes.csv"
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']

df = pandas.read_csv(filename, names=names)

array = df.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)

model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=kfold)

print(results.mean())

