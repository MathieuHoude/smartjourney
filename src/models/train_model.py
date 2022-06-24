from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
import pandas as pd
from sentence_transformers import SentenceTransformer
from numpy import mean, std

def train_inspection_predictions():
    inspection_per_elevator = pd.read_csv('./data/processed/inspection_per_elevator.csv')
    order = pd.read_csv('./data/processed/order.csv')
    X = inspection_per_elevator.iloc[:, :-1]
    y = inspection_per_elevator["CURRENT"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    #Sentences are encoded by calling model.encode()
    model.fit(X_train, y_train)
    order["EMBEDDINGS"] = model.encode(order["DIRECTIVEWITHINFORMATION"].to_list())
    print(order["EMBEDDINGS"])
    order.to_csv("./data/processed/order_with_embeddings.csv")

    print(X)
    transforms = list()
    # transforms.append(('mms', MinMaxScaler()))
    transforms.append(('ss', StandardScaler()))
    # transforms.append(('rs', RobustScaler()))
    transforms.append(('pca', PCA(n_components=7)))

    # create the feature union
    fu = FeatureUnion(transforms)
    model = LogisticRegression(solver='liblinear')
    steps = list()
    steps.append(('fu', fu))
    steps.append(('m', model))
    pipeline = Pipeline(steps=steps)

    # define the cross-validation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    scores = cross_val_score(pipeline, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
    # report performance
    print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
    print(scores)
    pipeline.fit(X_train, y_train)
    print(pipeline.score(X_test, y_test))

train_inspection_predictions()