from pathlib import Path
import pickle
import numpy as np
import xgboost
import optuna
from optuna.samplers import TPESampler
from optuna.integration import OptunaSearchCV
from optuna import distributions
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix, 
    ConfusionMatrixDisplay
)

def _stack_features(df, x_col):
    return np.asarray(np.vstack(df[x_col].to_numpy()), dtype=float)

def train_optuna_xgb(df, x_col, y_col, model_name, save_dir):
    X = _stack_features(df, x_col)
    y = df[y_col].to_numpy(dtype=int)

    num_zeros = (y == 0).sum()
    num_ones = (y == 1).sum()
    scw = num_zeros / num_ones

    xgb = xgboost.XGBClassifier(
        random_state=42,
        tree_method='hist',
        max_bin=256,
        n_jobs=8
    )

    params = {
        'n_estimators': distributions.IntDistribution(100, 1000),
        'max_depth': distributions.IntDistribution(1, 20),
        'learning_rate': distributions.FloatDistribution(1e-3, 0.1, log=True),
        'subsample': distributions.FloatDistribution(0.05, 1.0),
        'colsample_bytree': distributions.FloatDistribution(0.05, 1.0),
        'min_child_weight': distributions.IntDistribution(1, 20),
        'reg_alpha': distributions.FloatDistribution(1e-8, 1.0, log=True),
        'reg_lambda': distributions.FloatDistribution(1e-3, 10.0, log=True),
        'scale_pos_weight': distributions.FloatDistribution(scw / 2, scw * 2),
    }

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)

    optuna_search = OptunaSearchCV(
        estimator=xgb,
        param_distributions=params,
        scoring='roc_auc',
        cv=5,
        n_jobs=1,
        study=study,
        n_trials = 10
    )

    optuna_search.fit(X, y)

    best_clf = optuna_search.best_estimator_

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / f'{model_name}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(best_clf, f, protocol=4)

    print(optuna_search.best_score_)

    return best_clf, optuna_search.best_score_

def test(
    df,
    name,
    model_pkl_path=None,
    x_col='xtb_concat',
    y_col='cryst',
    ax=None,
    display_labels=('0', '1'),
):

    if model_pkl_path is None:
        model_pkl_path = f'best_classifier_{name}.pkl'

    with open(model_pkl_path, 'rb') as f:
        clf = pickle.load(f)

    X = np.vstack(df[x_col].to_numpy())
    y = df[y_col].to_numpy(dtype=int)
    pred = clf.predict(X)
    proba = clf.predict_proba(X)[:, 1]

    print(name)
    print(f'  F1        : {f1_score(y, pred):.3f}')
    print(f'  ROC-AUC   : {roc_auc_score(y, proba):.3f}')
    print(f'  Precision : {precision_score(y, pred):.3f}')
    print(f'  Recall    : {recall_score(y, pred):.3f}')

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(4, 4))

    disp = ConfusionMatrixDisplay(
        confusion_matrix(y, pred),
        display_labels=list(display_labels),
    )
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
    ax.set_title(name)

    if created:
        plt.tight_layout()
        plt.show()

    return disp
