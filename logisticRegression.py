from data.dataRead import get_train_test_data
from sklearn.linear_model import LogisticRegression

# obtenermos los datos para generar modelo
x_train, x_test, y_train, y_test = get_train_test_data()

# generar modelo
model = LogisticRegression()
model.fit(x_train, y_train)

# ver resultados del modelo
result = model.score(x_test, y_test)

print(result)

