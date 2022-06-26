import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer,InputExample, losses
# from src.models.train_model import features
from torch.utils.data import DataLoader
# def generate_dummies(df, columns):




def build_inspection_dataset():
    inspected = pd.read_csv('./data/interim/inspected.csv')
    order = pd.read_csv('./data/processed/order.csv')

    inspected["EARLIEST_INSPECTION_DATE"] = pd.to_datetime(inspected["EARLIEST_INSPECTION_DATE"])
    inspected.sort_values(by='EARLIEST_INSPECTION_DATE', inplace=True)

    threshold = 500
    outcomes = inspected["INSPECTIONOUTCOME"].value_counts()
    to_be_changed = []
    for outcome, count in outcomes.items():
        if count < threshold:
            to_be_changed.append(outcome)
    inspected.loc[inspected["INSPECTIONOUTCOME"].isin(to_be_changed), "INSPECTIONOUTCOME"] = "Other"
    inspected.to_csv('./data/processed/inspected.csv')

    inspectedWithDummies = pd.get_dummies(inspected, columns=["INSPECTIONTYPE", "INSPECTIONOUTCOME"])
    x = inspectedWithDummies.columns[inspectedWithDummies.columns.str.contains("INSPECTIONTYPE")].to_list()
    y = inspectedWithDummies.columns[inspectedWithDummies.columns.str.contains("INSPECTIONOUTCOME")].to_list()
    x.extend(y)
    x.append("ELEVATINGDEVICESNUMBER")
    x.append("INSPECTIONNUMBER")
    inspectedWithDummies = inspectedWithDummies[x]
    inspectedWithDummies = inspectedWithDummies.replace(0, np.nan)
    inspectedWithDummies.to_csv('./data/processed/inspectedWithDummies.csv')

    inspection_per_elevator = inspectedWithDummies.groupby("ELEVATINGDEVICESNUMBER").count()
    inspected_copy = inspected.copy()
    latestInspections = inspected_copy.groupby("ELEVATINGDEVICESNUMBER").first()

    inspection_without_latest = inspected_copy[~inspected_copy["INSPECTIONNUMBER"].isin(latestInspections["INSPECTIONNUMBER"])]
    dummies_inspection_without_latest = pd.get_dummies(inspection_without_latest, columns=["INSPECTIONTYPE", "INSPECTIONOUTCOME"])
    x = dummies_inspection_without_latest.columns[dummies_inspection_without_latest.columns.str.contains("INSPECTIONTYPE")].to_list()
    y = dummies_inspection_without_latest.columns[dummies_inspection_without_latest.columns.str.contains("INSPECTIONOUTCOME")].to_list()
    x.extend(y)
    x.append("ELEVATINGDEVICESNUMBER")
    # x.append("INSPECTIONNUMBER")
    dummies_inspection_without_latest = dummies_inspection_without_latest[x]
    dummies_inspection_without_latest = dummies_inspection_without_latest.replace(0, np.nan)
    inspection_per_elevator = dummies_inspection_without_latest.groupby("ELEVATINGDEVICESNUMBER").count()
    inspection_count_per_elevator = inspection_without_latest.groupby("ELEVATINGDEVICESNUMBER").count().reset_index()
    inspection_per_elevator['CURRENT'] = inspection_count_per_elevator['ELEVATINGDEVICESNUMBER'].map(latestInspections['INSPECTIONOUTCOME'])
    inspection_per_elevator = inspection_per_elevator.dropna()
    inspection_per_elevator['CURRENT'] =  pd.factorize( inspection_per_elevator['CURRENT'] )[0]

    inspection_per_elevator['DIRECTIVEWITHINFORMATION'] = order['DIRECTIVEWITHINFORMATION']
    inspection_per_elevator['DIRECTIVEWITHINFORMATION'] = inspection_per_elevator['DIRECTIVEWITHINFORMATION'].fillna(" ")
    inspection_per_elevator.to_csv('./data/processed/inspection_per_elevator.csv')

def build_order_dataset():
    order = pd.read_csv("./data/raw/order.csv")
    order.columns = order.columns.str.upper().str.replace(' ', '')
    order["DATEOFISSUE"] = pd.to_datetime(order["DATEOFISSUE"])
    order.sort_values(by='DATEOFISSUE', inplace=True)
    order[["DIRECTIVE", "INSPECTIONSADDITIONALINFORMATION"]] = order[["DIRECTIVE", "INSPECTIONSADDITIONALINFORMATION"]].fillna(" ")
    order["DIRECTIVEWITHINFORMATION"] = order[["DIRECTIVE", "INSPECTIONSADDITIONALINFORMATION"]].agg(' '.join,axis=1)

    order["RISKSCORE"] = order["RISKSCORE"].fillna(0)
    
    order.to_csv("./data/processed/order.csv")

# def generate_embeddings():
#     inspection_per_elevator = pd.read_csv('./data/processed/inspection_per_elevator.csv')
#     X = inspection_per_elevator[features]
#     y = inspection_per_elevator["CURRENT"]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

#     column = list(X_train["DIRECTIVEWITHINFORMATION"])
#     train_examples = []
#     for e in column:
#         train_examples.append(InputExample(texts=[e]))

#     #Define your train examples. You need more than just two examples...
#     # train_examples = [InputExample(texts=X_train["DIRECTIVEWITHINFORMATION"], label=0.8)]

#     model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
#     #Sentences are encoded by calling model.encode()
#     train_dataloader = DataLoader(train_examples, shuffle=False, batch_size=16)
#     train_loss = losses.BatchAllTripletLoss(model)
#     model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
#     inspection_per_elevator["EMBEDDINGS"] = model.encode(inspection_per_elevator["DIRECTIVEWITHINFORMATION"].to_list())
#     print(inspection_per_elevator["EMBEDDINGS"])
#     inspection_per_elevator.to_csv("./data/processed/order_with_embeddings.csv")


build_order_dataset()
build_inspection_dataset()
