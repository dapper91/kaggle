import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.feature_selection import RFE
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC


def prepare_features(df):
    df['Age'].fillna(df['Age'].mean(), inplace = True)
    df['Fare'].fillna(df['Fare'].mean(), inplace = True)

    df['Sex'] = label_binarize(df['Sex'], classes = ['males', 'female'])
    
    # disabled as usefull transformation
    # df['Fare'] = df['Fare'].apply(lambda x: int(round(math.log(x+1))))

    df_embarked = pd.DataFrame(label_binarize(df['Embarked'], classes = ['C', 'Q', 'S']), columns = ['Embarked_C',
                                                                                                     'Embarked_Q',
                                                                                                     'Embarked_S'])

    df = pd.concat([df, df_embarked], axis = 1, copy = False)

    return df



# ----- PREPROCESSING -----

df_train = prepare_features(pd.read_csv('train.csv'))
df_test  = prepare_features(pd.read_csv('test.csv'))


# test for NaN value existence
# for col in df_train:
# 	print("%s: %d" % (col, any(df_train[col].isnull())))


#classifier = SVC()
classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', max_depth = 8)



# ----- FEATURE SELECTION -----

# # all features:
# features, output = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S'], 'Survived'
# X_train, y_train = df_train[features], df_train[output]
# X_test = df_test[features]

# # selecting best four features:
# selector = RFE(classifier, 4, step = 1)
# selector = selector.fit(X_train, y_train)
# print("best features: " + ", ".join([feature for feature, rank in zip(features, selector.ranking_) if rank == 1]))



features, output = ['Pclass', 'Sex', 'Age', 'Fare'], 'Survived'

X_train, y_train = df_train[features], df_train[output]
X_test = df_test[features]



# ----- GRID_SEARCH -----

# parameters = {
# 	'n_estimators': [10, 30, 50, 100, 200], 
# 	'max_depth': 	[5, 8, 10, 20]
# }

# classifier = GridSearchCV(classifier, parameters)
# classifier.fit(X_train, y_train)
# print("best score: %f", classifier.best_score_)
# print("best params: ")
# for param in classifier.best_params_:
# 	print("%s: %d" % (param, classifier.best_params_[param]))



# ----- TRAINING AND VALIDATION -----

print("Training...")

classifier.fit(X_train, y_train)
score = cross_val_score(classifier, X_train, y_train, cv = KFold(len(X_train), n_folds = 10, shuffle = False))

# according to myfirstforest as baseline
# base score: about 0.81
print("baseline score: %.3f" % 0.81)
print("current score:  %.3f" % score.mean())



print("Predicting...")
output = classifier.predict(X_test)

df_predicted = pd.DataFrame({'Survived': output}, index = df_test['PassengerId'])
df_predicted.to_csv('result_1.csv')


print("Done.")