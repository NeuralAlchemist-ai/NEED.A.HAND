from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import optuna


class Objective:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __call__(self, trial):
        classifier_name = trial.suggest_categorical("classifier", ["SVC", "RandomForest"])

        if classifier_name == "SVC":
            svc_c = trial.suggest_float("svc_c", 1e-3, 1e3, log=True)
            svc_kernel = trial.suggest_categorical("svc_kernel", ["rbf", "poly"])
            svc_gamma = trial.suggest_float("svc_c", 1e-5, 1e0, log=True)
            if svc_kernel == "poly":
                svc_degree = trial.suggest_int("degree", 2, 4)
            else:
                svc_degree = 3

            model = SVC(
                    C=svc_c,
                    kernel=svc_kernel,
                    gamma=svc_gamma,
                    degree=svc_degree,
                    random_state=42
            )

        else:
            rf_n_estimators = trial.suggest_int("rf_n_estimators", 100, 800)
            rf_max_depth = trial.suggest_int("rf_max_depth", 5, 40)
            rf_min_samples_split = trial.suggest_int("rf_min_samples_split", 2, 20)
            rf_min_samples_leaf = trial.suggest_int("rf_min_samples_leaf", 1, 10)
            rf_max_features = trial.suggest_categorical("rf_max_features", ["sqrt", "log2", 0.3, 0.5])

            model = RandomForestClassifier(
                    n_estimators=rf_n_estimators,
                    max_depth=rf_max_depth,
                    min_samples_split=rf_min_samples_split,
                    min_samples_leaf=rf_min_samples_leaf,
                    max_features=rf_max_features,
                    random_state=42
            )

        return cross_val_score(model, self.X, self.y, cv=5, scoring="accuracy").mean()


study_svc = optuna.create_study(direction="maximize")
study_rf = optuna.create_study(direction="maximize")