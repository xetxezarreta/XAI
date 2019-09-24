import pandas
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# leer datos
filename = "data/pima-indians-diabetes.csv"
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
df = pandas.read_csv(filename, names=names)

# preparar datos
array = df.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,random_state=seed)

# generar modelo
model = LogisticRegression()
model.fit(X_train, Y_train)

# ver resultados del modelo
result = model.score(X_test, Y_test)

print(result)

