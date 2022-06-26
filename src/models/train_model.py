from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
import pandas as pd
from sentence_transformers import SentenceTransformer,InputExample, losses
from numpy import column_stack, mean, std
from torch.utils.data import DataLoader

features = ['ELEVATINGDEVICESNUMBER', 'INSPECTIONTYPE_ED-Enforcement Action',
       'INSPECTIONTYPE_ED-FU Enforcement Action Insp',
       'INSPECTIONTYPE_ED-Followup Inspection',
       'INSPECTIONTYPE_ED-Followup Lic Insp',
       'INSPECTIONTYPE_ED-Followup Minor Alt',
       'INSPECTIONTYPE_ED-Followup No-Lic Insp',
       'INSPECTIONTYPE_ED-Followup Ownership Change',
       'INSPECTIONTYPE_ED-Followup Reg Non-Compliance',
       'INSPECTIONTYPE_ED-Initial Inspection',
       'INSPECTIONTYPE_ED-Inspection Temp Lic',
       'INSPECTIONTYPE_ED-MCP Enforcement Insp',
       'INSPECTIONTYPE_ED-MCP Follow up',
       'INSPECTIONTYPE_ED-Major Alteration Inspection',
       'INSPECTIONTYPE_ED-Minor A Inspection',
       'INSPECTIONTYPE_ED-Minor B Inspection',
       'INSPECTIONTYPE_ED-Non-Mandated Followup ON',
       'INSPECTIONTYPE_ED-Non-Mandated Insp ON',
       'INSPECTIONTYPE_ED-PWGSC Insp',
       'INSPECTIONTYPE_ED-Perform L1 Incident Insp',
       'INSPECTIONTYPE_ED-Periodic Inspection',
       'INSPECTIONTYPE_ED-Re-Activate Inspection',
       'INSPECTIONTYPE_ED-Reg Non-Compliance',
       'INSPECTIONTYPE_ED-Sub  Inspection',
       'INSPECTIONTYPE_ED-Sub Failed Initial',
       'INSPECTIONTYPE_ED-Sub Inspection',
       'INSPECTIONTYPE_ED-Sub Inspection Major',
       'INSPECTIONTYPE_ED-Unscheduled Inspection',
       'INSPECTIONOUTCOME_All Orders Resolved', 'INSPECTIONOUTCOME_Complete',
       'INSPECTIONOUTCOME_DC Follow up', 'INSPECTIONOUTCOME_Fail Initial',
       'INSPECTIONOUTCOME_Follow Up Initial', 'INSPECTIONOUTCOME_Follow up',
       'INSPECTIONOUTCOME_Follow up Major',
       'INSPECTIONOUTCOME_Follow up Sub Major', 'INSPECTIONOUTCOME_Other',
       'INSPECTIONOUTCOME_Passed', 'INSPECTIONOUTCOME_Passed Major',
       'INSPECTIONOUTCOME_Shutdown', 'INSPECTIONOUTCOME_Unable to Inspect',
       'CURRENT']

def generate_embeddings():
    inspection_per_elevator = pd.read_csv('./data/processed/inspection_per_elevator.csv')
    X = inspection_per_elevator['DIRECTIVEWITHINFORMATION']
    y = inspection_per_elevator["CURRENT"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

    train_sentenceTransformer(list(X_train["DIRECTIVEWITHINFORMATION"]))

    model = SentenceTransformer('./models/')
    sentence_embeddings  = model.encode(inspection_per_elevator["DIRECTIVEWITHINFORMATION"].to_list())
    inspection_per_elevator = inspection_per_elevator.join(pd.DataFrame(sentence_embeddings).add_prefix("EMBEDDING - "))
    print(inspection_per_elevator)
    inspection_per_elevator.to_csv("./data/processed/order_with_embeddings.csv")

    

def train_sentenceTransformer(sentences):
    train_examples = []
    for e in sentences:
        train_examples.append(InputExample(texts=[e]))

    #Define your train examples. You need more than just two examples...
    # train_examples = [InputExample(texts=X_train["DIRECTIVEWITHINFORMATION"], label=0.8)]

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    #Sentences are encoded by calling model.encode()
    train_dataloader = DataLoader(train_examples, shuffle=False, batch_size=16)
    train_loss = losses.BatchAllTripletLoss(model)
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100, output_path="./models/",checkpoint_path="./models/checkpoints/")

def train_inspection_predictions():
    inspection_per_elevator = pd.read_csv('./data/processed/order_with_embeddings.csv')

    embedding_columns = [col for col in inspection_per_elevator.columns if 'EMBEDDING' in col]
    _features = features + embedding_columns
    X = inspection_per_elevator[_features]
    y = inspection_per_elevator["CURRENT"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

    transforms = list()
    # transforms.append(('mms', MinMaxScaler()))
    transforms.append(('ss', StandardScaler()))
    # transforms.append(('rs', RobustScaler()))
    #transforms.append(('pca', PCA(n_components=7)))

    # create the feature union
    fu = FeatureUnion(transforms)
    model = LogisticRegression(solver='newton-cg')
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

#generate_embeddings()
train_inspection_predictions()
