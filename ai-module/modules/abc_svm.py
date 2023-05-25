import numpy as np
import matplotlib.pyplot as plt
import csv

from sklearn import svm
from sklearn import metrics
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split

from joblib import dump, load

class AbcSVM:

    def prepare_svm(self, dataset):

        X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=109)
        clf = svm.SVC(kernel='linear')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        print(y_test, y_pred)
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        print("Precision:",metrics.precision_score(y_test, y_pred,average='micro'))
        print("Recall:",metrics.recall_score(y_test, y_pred,average='micro'))
        self.make_plot(clf, X_test, y_test)

    # Метод для отладки модели, на проде можно не использовать
    def make_plot(self, svm, X_test, y_test):
        h = 1

        x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
        y_min, y_max = X_test[:, 1].min(), X_test[:, 1].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        X_hypo = np.c_[xx.ravel().astype(np.float32),
                    yy.ravel().astype(np.float32)]

        zz = svm.predict(X_hypo)
        zz = zz.reshape(xx.shape)

        print(xx, yy, zz)

        plt.style.use('ggplot')
        plt.set_cmap('jet')
        plt.xlabel(('x'))
        plt.ylabel('y')
        plt.contourf(xx, yy, zz, cmap=plt.cm.coolwarm, alpha=0.8)
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=50)
        plt.savefig('test.png')


    def load_dataset(self, dataset_csv):

        with open(dataset_csv) as csv_file:
            data_reader = csv.reader(csv_file, delimiter=';')
            feature_names = next(data_reader)[:-1]
            data = []
            target = []

            for row in data_reader:
                features = row[:-1]
                label = row[-1]
                data.append([float(num) for num in features])
                target.append(int(label))
            
            data = np.array(data)
            target = np.array(target)

        return Bunch(data=data, target=target, feature_names=feature_names)
