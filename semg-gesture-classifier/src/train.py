import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC


class SVCObjective:

  def __init__(self, X, y):
    self.X = X
    self.y = y

  def __call__(self, trial):
    svc_c = trial.suggest_float("svc_c", 1e-3, 1e3, log=True)
    svc_kernel = trial.suggest_categorical("svc_kernel", ["rbf", "poly"])
    svc_gamma = trial.suggest_float("svc_gamma", 1e-5, 1e0, log=True)
    svc_degree = (
        trial.suggest_int("degree", 2, 4) if svc_kernel == "poly" else 3
    )

    model = SVC(
        C=svc_c,
        kernel=svc_kernel,
        gamma=svc_gamma,
        degree=svc_degree,
        random_state=42,
    )
    return cross_val_score(
        model, self.X, self.y, cv=5, scoring="f1_macro"
    ).mean()


class RandomForestObjective:

  def __init__(self, X, y):
    self.X = X
    self.y = y

  def __call__(self, trial):
    rf_n_estimators = trial.suggest_int("rf_n_estimators", 100, 800)
    rf_max_depth = trial.suggest_int("rf_max_depth", 5, 40)
    rf_min_samples_split = trial.suggest_int("rf_min_samples_split", 2, 20)
    rf_min_samples_leaf = trial.suggest_int("rf_min_samples_leaf", 1, 10)
    rf_max_features = trial.suggest_categorical(
        "rf_max_features", ["sqrt", "log2", 0.3, 0.5]
    )

    model = RandomForestClassifier(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        min_samples_split=rf_min_samples_split,
        min_samples_leaf=rf_min_samples_leaf,
        max_features=rf_max_features,
        random_state=42,
    )
    return cross_val_score(
        model, self.X, self.y, cv=5, scoring="f1_macro"
    ).mean()

# 0. Init preprocessed data
X_train, X_test, y_train, y_test = ..., ..., ..., ...

# 1. Optimize SVC on training data
svc_study = optuna.create_study(direction="maximize")
svc_study.optimize(SVCObjective(X_train, y_train), n_trials=50)

# 2. Optimize Random Forest on training data
rf_study = optuna.create_study(direction="maximize")
rf_study.optimize(RandomForestObjective(X_train, y_train), n_trials=50)

# 3. Build and fit the winning model
if svc_study.best_value >= rf_study.best_value:
  best_model_name = "SVC"
  best_params = svc_study.best_params
  best_model = SVC(
      C=best_params["svc_c"],
      kernel=best_params["svc_kernel"],
      gamma=best_params["svc_gamma"],
      degree=best_params.get("degree", 3),
      random_state=42,
  )
else:
  best_model_name = "RandomForest"
  best_params = rf_study.best_params
  best_model = RandomForestClassifier(
      n_estimators=best_params["rf_n_estimators"],
      max_depth=best_params["rf_max_depth"],
      min_samples_split=best_params["rf_min_samples_split"],
      min_samples_leaf=best_params["rf_min_samples_leaf"],
      max_features=best_params["rf_max_features"],
      random_state=42,
  )

best_model.fit(X_train, y_train)

# 4. Evaluation Block on Hold-Out Test Set
y_pred = best_model.predict(X_test)

macro_f1 = f1_score(y_test, y_pred, average="macro")
balanced_acc = balanced_accuracy_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

print(f"--- Evaluation Results ({best_model_name}) ---")
print(f"Macro F1-Score:    {macro_f1:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"MCC Score:         {mcc:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))