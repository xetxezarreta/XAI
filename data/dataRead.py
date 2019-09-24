import pandas
from sklearn.model_selection import train_test_split


def get_train_test_data():
    # obtener los datos
    filename = "data/pima-indians-diabetes.csv"
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    df = pandas.read_csv(filename, names=names)

    # preparar los datos
    array = df.values
    X = array[:, 0:8]
    Y = array[:, 8]
    test_size = 0.33
    seed = 7
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    return x_train, x_test, y_train, y_test
