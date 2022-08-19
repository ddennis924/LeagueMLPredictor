
from tkinter import Y
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier

def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation

    Parameters
    ----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data

    Returns
    ----------
        pandas Series with mean scores from cross_validation
    """

    scores = cross_validate(model, X_train, y_train, **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append((f"%0.5f (+/- %0.5f)" % (mean_scores[i], std_scores[i])))

    return pd.Series(data=out_col, index=mean_scores.index)

df = pd.read_csv("LeagueMatches.csv", index_col=0)
df = df.replace(-1, np.nan)

train_df, test_df = train_test_split(df, test_size=0.3, random_state=123)

X_train = train_df.drop(columns='winner')
y_train = train_df['winner']
X_test = test_df.drop(columns='winner')
y_test = test_df['winner']
cat_feats = ['blueTops', 'blueJngs', 'blueMids', 'blueBots', 'blueSups', 'redTops', 'redJngs', 'redMids', 'redBots', 'redSups']
num_feats = ['btmp', 'btwr', 'btt', 'bjmp', 'bjwr', 'bjt', 'bmmp', 'bmwr', 'bmt', 'bbmp', 'bbwr', 'bbt', 'bsmp', 'bswr', 'bst', 'rtmp', 'rtwr', 'rtt', 'rjmp', 'rjwr', 'rjt', 'rmmp', 'rmwr', 'rmt', 'rbmp', 'rbwr', 'rbt', 'rsmp', 'rswr','rst']

preprocessor = make_column_transformer((OneHotEncoder(handle_unknown='ignore'), cat_feats), 
(make_pipeline(SimpleImputer(strategy='median'),StandardScaler()), num_feats))

dummy = make_pipeline(preprocessor, DummyClassifier())
cross_val_results_dummy = pd.DataFrame(
    cross_validate(dummy, X_train, y_train, return_train_score=True)
)
print(cross_val_results_dummy.head())

pipe_lr = make_pipeline(preprocessor, LogisticRegression())
cross_val_results_lr = pd.DataFrame(
    cross_validate(pipe_lr, X_train, y_train, return_train_score=True)
)
print(cross_val_results_lr)

# param_grid = {"randomforestclassifier__max_depth": [1, 10, 50, 100],
#               "randomforestclassifier__n_estimators":[1, 10, 50, 100, 200], 
#               "randomforestclassifier__max_features":[1, 10, 20, 50, 100]}
param_grid = {
    "svc__gamma": [0.001, 0.01, 0.1, 1.0 ,10, 100],
    "svc__C":[0.001, 0.01, 0.1, 1.0, 10, 100]
}
pipe_svc = make_pipeline(preprocessor, SVC())
rand_search = GridSearchCV(pipe_svc, param_grid=param_grid, n_jobs=-1, cv=10)


rand_search.fit(X_train, y_train)
print("best params for accuracy:")
print(rand_search.best_params_)
print("best score for accuracy:")
print(rand_search.best_score_)
pipe_svc = make_pipeline(preprocessor, SVC(C=10, gamma=0.1, probability=True))
cross_val_results_svc = pd.DataFrame(
    cross_validate(pipe_svc, X_train, y_train, return_train_score=True)
)
print("svc")
print(cross_val_results_svc.to_string())
pipe_svc.fit(X_train, y_train)
print(pipe_svc.score(X_test, y_test))



pipe_lr = make_pipeline(preprocessor, LogisticRegression())
param_grid = {
    "logisticregression__C":[0.001, 0.01, 0.1, 1.0, 10, 100, 1000]
}
rand_search_lr = GridSearchCV(pipe_lr, param_grid=param_grid, n_jobs=-1, cv=10)

rand_search_lr.fit(X_train, y_train)
print("best params for accuracy:")
print(rand_search_lr.best_params_)
print("best score for accuracy:")
print(rand_search_lr.best_score_)
pipe_lr = make_pipeline(preprocessor, LogisticRegression(C=1.0))
cross_val_results_lr = pd.DataFrame(
    cross_validate(pipe_lr, X_train, y_train, return_train_score=True)
)
print("lr")
print(cross_val_results_lr.to_string())
pipe_lr.fit(X_train, y_train)
print(pipe_lr.score(X_test, y_test))

classifier = {"logisticregression": pipe_lr, "svc": pipe_svc}
average_model = VotingClassifier(list(classifier.items()), voting='soft')
cross_val_results_avg = pd.DataFrame(
    cross_validate(average_model, X_train, y_train, return_train_score=True)
)
print(cross_val_results_avg.to_string())
average_model.fit(X_train, y_train)
print(average_model.score(X_test, y_test))

# from xgboost import XGBClassifier
# from lightgbm.sklearn import LGBMClassifier

# models = {
#     "SVC": make_pipeline(preprocessor, SVC(class_weight="balanced")),
#     "Random Forest": make_pipeline(preprocessor, RandomForestClassifier(class_weight="balanced")),
#     "XGB": make_pipeline(
#     preprocessor, XGBClassifier(random_state=123, eval_metric="logloss", verbosity=0,use_label_encoder=False, class_weight="balanced")),
#     "LightGBM": make_pipeline(preprocessor, LGBMClassifier(random_state=123, class_weight="balanced")),
# }

# #imported from lecutre 11
# results = {}
# for (name, model) in models.items():
#     results[name] = mean_std_cross_val_scores(
#         model, X_train, y_train, return_train_score=True, scoring=scores
#     )
# pd.DataFrame(results).T
