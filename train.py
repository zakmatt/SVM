from data import Data
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from numpy.random import RandomState as random_state
from sklearn.model_selection import train_test_split as data_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC


def scale_features(x_train, x_test):
    scaller = StandardScaler()
    x_train = scaller.fit_transform(x_train)
    x_test = scaller.transform(x_test)
    return x_train, x_test


def visualize(x, y, classifier):
    x_1, x_2 = np.meshgrid(
        np.arange(start=x[:, 0].min() - 1, stop=x[:, 0].max() + 1, step=0.01),
        np.arange(start=x[:, 1].min() - 1, stop=x[:, 1].max() + 1, step=0.01)
    )
    plt.contourf(x_1,
                 x_2,
                 classifier.predict(np.array([x_1.ravel(), x_2.ravel()]).T).reshape(x_1.shape),
                 alpha=0.75,
                 cmap=ListedColormap(('red', 'green')))
    plt.xlim(x_1.min(), x_1.max())
    plt.xlim(x_2.min(), x_2.max())

    for i, j in enumerate(np.unique(y)):
        plt.scatter(
            x[y == j, 0], x[y == j, 1],
            c=ListedColormap(('red', 'green'))(i),
            label=j
        )
    plt.title('SVM')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    data = Data('data/Social_Network_Ads.csv')
    x = data.get_data(col=2, end_col=3, columns=True).astype(np.float32)
    y = data.get_data(col=4).astype(np.float32)

    x_train, x_test, y_train, y_test = data_split(x, y, test_size=0.25, random_state=random_state())
    x_train, x_test = scale_features(x_train, x_test)

    classifier = SVC(kernel='linear', random_state=random_state())

    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    conf_mtx = confusion_matrix(y_test, y_test)
    
    visualize(x_train, y_train, classifier)