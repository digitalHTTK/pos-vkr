import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

from sklearn import tree

class TreeOcupancy:

    def one_hot_encode(self, X):
        column_transformer = make_column_transformer(
            (OneHotEncoder(), ['weekday', 'weather']),
            remainder='passthrough')
        encoded = column_transformer.fit_transform(X).toarray()
        return pd.DataFrame(data=encoded, columns=column_transformer.get_feature_names_out())
    
    def get_features(self, X):
        column_transformer = make_column_transformer(
            (OneHotEncoder(), ['weekday', 'weather']),
            remainder='passthrough')
        column_transformer.fit_transform(X).toarray()
        return column_transformer.get_feature_names_out()

    def save_png_model(self, clf, features):
        fig = plt.figure(figsize=(25,20))
        _ = tree.plot_tree(clf, 
                        feature_names=features,  
                        class_names=['0','1'],
                        filled=True)
        fig.savefig("decistion_tree.png")

    def train_model(self):
        X = pd.read_csv('tree_o.csv',usecols=['weekday', 'weather', 'is_occup'],delimiter=';')
        X = X.dropna()
        y = X.pop('is_occup')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 100)
        
        X_train_ = X_train
        X_train = self.one_hot_encode(X_train)
        X_test = self.one_hot_encode(X_test)

        clf = DecisionTreeClassifier(max_depth=4, criterion='entropy', max_features=0.6, splitter='best')
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        print(accuracy_score(y_test, predictions))
        self.save_png_model(clf, self.get_features(X_train_))
        