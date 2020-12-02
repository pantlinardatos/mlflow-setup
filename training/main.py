import mlflow
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    raw = load_iris()
    data = raw['data']
    labels = raw['target']

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=10)

    with mlflow.start_run() as run:
        clf = LogisticRegression()
        hyper_parameters = clf.get_params()
        mlflow.log_params(hyper_parameters)

        clf.fit(x_train, y_train)
        predictions = clf.predict(x_test)
        acc = accuracy_score(y_test, predictions)
        mlflow.log_metric("acc", acc)
