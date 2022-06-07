import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score


def inverse_propensity_weighting(data: pd.DataFrame):
    features = data.drop(columns=["T", "Y"])
    treatments = data["T"]
    results = data["Y"]

    model = LogisticRegression(max_iter=1e5)

    model.fit(features, treatments)
    probabilities = model.predict_proba(features)

    propensity_weighting = probabilities[:, 1] / probabilities[:, 0]

    treatment_mask = treatments == 1
    treated_weight = np.sum(results[treatment_mask]) / np.sum(treatments)
    untreated_weight = np.sum(results[~treatment_mask] * propensity_weighting[~treatment_mask]) \
                       / np.sum(propensity_weighting[~treatment_mask])

    ATT = treated_weight - untreated_weight
    return ATT


def main():
    path_data_1 = r"data/data1.csv"
    path_data_2 = r"data/data2.csv"

    df_1 = pd.read_csv(path_data_1)
    df_2 = pd.read_csv(path_data_2)

    df_1 = pd.get_dummies(df_1)
    df_2 = pd.get_dummies(df_2)

    print(f"IPW ATT: {inverse_propensity_weighting(df_1)}")
    print(f"IPW ATT: {inverse_propensity_weighting(df_2)}")


if __name__ == '__main__':
    main()
