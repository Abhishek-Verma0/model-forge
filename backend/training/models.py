"""The training model table: what each model is, what data it accepts, how it is
built, which parameters are editable and whether it trains in epochs. Capability
flags are facts measured on the installed versions (2026-09-13/14); the self-check
refits every row on four data shapes and fails if a flag is wrong. XGBoost /
LightGBM / CatBoost rows appear only when the library imports and a tiny fit works.

Device: a model trains on the GPU whenever its library can use one here (checked with
a tiny GPU fit); the CPU is used only when there is no usable GPU. Measured 2026-09-15
(RTX 5070 Laptop): only XGBoost and CatBoost can -- the installed LightGBM wheel has no
GPU build and scikit-learn models have no GPU training.

`epochs` names the adapter in training/epochs.py (None = fits in one call).
Parameter defaults come from the library (training/params.py); ranges are ours.
"""

import json
from functools import lru_cache

import numpy as np
from sklearn import dummy, ensemble, linear_model, naive_bayes, neighbors, neural_network, svm, tree

from training.params import AUTO, fit_specs, spec

LOGREG_MAX_ITER = 1000   # our starting point: scikit-learn's default 100 often stops before converging
_CW = {"class_weight": "balanced"}
_SVM_NOTE = "fit time grows ~4x per doubling of rows (measured)"
_KNN_NOTE = "distances lose meaning on wide text data"
_MLP_NOTE = "first layer has features x units weights - slow and memory-heavy on wide text data"


def _row(label, family, make, sparse=True, nonneg=False, nan=False, balanced=None, note=None,
         params=(), epochs=None, resolve=None, device="cpu"):
    return {"label": label, "family": family, "make": make, "sparse": sparse, "nonneg": nonneg, "nan": nan,
            "balanced": balanced, "note": note, "params": list(params), "epochs": epochs, "resolve": resolve,
            "device": device}


# parameter specs reused across rows (ranges: our starting point)
def _lr(hi=1.0):
    return spec("learning_rate", "Learning rate", "float", 1e-4, hi, log=True)


_TREES = [spec("n_estimators", "Trees", "int", 1, 5000),
          spec("max_depth", "Max depth", "int", 1, 200, none=True, help="none = grow until leaves are pure"),
          spec("min_samples_leaf", "Min rows per leaf", "int", 1, 10000)]
_TREE = _TREES[1:]
_KNN = [spec("n_neighbors", "Neighbours (k)", "int", 1, 500),
        spec("weights", "Weights", "choice", choices=["uniform", "distance"])]
_SVM = [spec("C", "C (inverse regularisation)", "float", 1e-4, 1e4, log=True),
        spec("kernel", "Kernel", "choice", choices=["linear", "poly", "rbf", "sigmoid"])]
_GB = [spec("n_estimators", "Boosting stages (epochs)", "int", 1, 5000), _lr(),
       spec("max_depth", "Max depth", "int", 1, 50)]
_HGB = [spec("max_iter", "Boosting rounds (epochs)", "int", 1, 5000), _lr(),
        spec("max_depth", "Max depth", "int", 1, 200, none=True)] + fit_specs()
_MLP = [spec("hidden_layer_sizes", "Hidden layers", "layers", help="units per layer, e.g. 64,32"),
        spec("max_iter", "Epochs", "int", 1, 10000),
        spec("learning_rate_init", "Learning rate", "float", 1e-6, 1.0, log=True),
        spec("alpha", "L2 penalty", "float", 0.0, 10.0),
        spec("batch_size", "Batch size", "int", 1, 100000, auto=True, help="auto = min(200, rows)")] + fit_specs()
_XGB = [spec("n_estimators", "Boosting rounds (epochs)", "int", 1, 10000), _lr(),
        spec("max_depth", "Max depth", "int", 1, 30)] + fit_specs()
_LGBM = [spec("n_estimators", "Boosting rounds (epochs)", "int", 1, 10000), _lr(),
         spec("num_leaves", "Leaves per tree", "int", 2, 131072)] + fit_specs()
_CAT = [spec("iterations", "Boosting rounds (epochs)", "int", 1, 10000),
        spec("learning_rate", "Learning rate", "float", 1e-4, 1.0, log=True, auto=True,
             help="auto = CatBoost chooses from data size"),
        spec("depth", "Tree depth", "int", 1, 16)] + fit_specs()


def _classifiers():
    return {
        "baseline": _row("Baseline (most frequent class)", "Baseline",
                         lambda seed: dummy.DummyClassifier(strategy="most_frequent"), nan=True),
        "logistic_regression": _row(
            "Logistic regression", "Linear",
            lambda seed: linear_model.LogisticRegression(max_iter=LOGREG_MAX_ITER, random_state=seed), balanced=_CW,
            params=[spec("C", "C (inverse regularisation)", "float", 1e-4, 1e4, log=True),
                    spec("max_iter", "Max iterations", "int", 10, 100000)]),
        "gaussian_nb": _row("Gaussian naive Bayes", "Naive Bayes", lambda seed: naive_bayes.GaussianNB(), sparse=False,
                            params=[spec("var_smoothing", "Variance smoothing", "float", 1e-12, 1.0, log=True)]),
        "multinomial_nb": _row("Multinomial naive Bayes", "Naive Bayes",
                               lambda seed: naive_bayes.MultinomialNB(), nonneg=True,
                               params=[spec("alpha", "Smoothing (alpha)", "float", 0.0, 100.0)]),
        "complement_nb": _row("Complement naive Bayes", "Naive Bayes",
                              lambda seed: naive_bayes.ComplementNB(), nonneg=True,
                              params=[spec("alpha", "Smoothing (alpha)", "float", 0.0, 100.0)]),
        "knn": _row("k-nearest neighbours", "Neighbours", lambda seed: neighbors.KNeighborsClassifier(),
                    note=_KNN_NOTE, params=_KNN),
        "decision_tree": _row("Decision tree", "Trees",
                              lambda seed: tree.DecisionTreeClassifier(random_state=seed), nan=True, balanced=_CW,
                              params=_TREE),
        "random_forest": _row("Random forest", "Trees",
                              lambda seed: ensemble.RandomForestClassifier(random_state=seed), nan=True, balanced=_CW,
                              params=_TREES),
        "extra_trees": _row("Extra trees", "Trees",
                            lambda seed: ensemble.ExtraTreesClassifier(random_state=seed), nan=True, balanced=_CW,
                            params=_TREES),
        "adaboost": _row("AdaBoost", "Boosting", lambda seed: ensemble.AdaBoostClassifier(random_state=seed),
                         params=[spec("n_estimators", "Estimators", "int", 1, 5000), _lr(10.0)]),
        "gradient_boosting": _row("Gradient boosting", "Boosting",
                                  lambda seed: ensemble.GradientBoostingClassifier(random_state=seed),
                                  params=_GB, epochs="gradient_boosting"),
        "hist_gradient_boosting": _row(
            "Histogram gradient boosting", "Boosting",
            lambda seed: ensemble.HistGradientBoostingClassifier(random_state=seed), sparse=False, nan=True,
            balanced=_CW, params=_HGB, epochs="hist_gradient_boosting"),
        "svm": _row("Support vector machine", "SVM", lambda seed: svm.SVC(random_state=seed),
                    balanced=_CW, note=_SVM_NOTE, params=_SVM),
        "mlp": _row("Neural network (MLP)", "Neural network",
                    lambda seed: neural_network.MLPClassifier(random_state=seed),
                    note=_MLP_NOTE, params=_MLP, epochs="mlp"),
    }


def _regressors():
    return {
        "baseline": _row("Baseline (training mean)", "Baseline",
                         lambda seed: dummy.DummyRegressor(strategy="mean"), nan=True),
        "linear_regression": _row("Linear regression", "Linear", lambda seed: linear_model.LinearRegression()),
        "ridge": _row("Ridge regression", "Linear", lambda seed: linear_model.Ridge(random_state=seed),
                      params=[spec("alpha", "Regularisation (alpha)", "float", 0.0, 1e6)]),
        "lasso": _row("Lasso regression", "Linear", lambda seed: linear_model.Lasso(random_state=seed),
                      params=[spec("alpha", "Regularisation (alpha)", "float", 1e-6, 1e6, log=True),
                              spec("max_iter", "Max iterations", "int", 10, 100000)]),
        "elastic_net": _row("Elastic net", "Linear", lambda seed: linear_model.ElasticNet(random_state=seed),
                            params=[spec("alpha", "Regularisation (alpha)", "float", 1e-6, 1e6, log=True),
                                    spec("l1_ratio", "L1 ratio", "float", 0.0, 1.0),
                                    spec("max_iter", "Max iterations", "int", 10, 100000)]),
        "knn": _row("k-nearest neighbours", "Neighbours", lambda seed: neighbors.KNeighborsRegressor(),
                    note=_KNN_NOTE, params=_KNN),
        "decision_tree": _row("Decision tree", "Trees",
                              lambda seed: tree.DecisionTreeRegressor(random_state=seed), nan=True, params=_TREE),
        "random_forest": _row("Random forest", "Trees",
                              lambda seed: ensemble.RandomForestRegressor(random_state=seed), nan=True, params=_TREES),
        "gradient_boosting": _row("Gradient boosting", "Boosting",
                                  lambda seed: ensemble.GradientBoostingRegressor(random_state=seed),
                                  params=_GB, epochs="gradient_boosting"),
        "hist_gradient_boosting": _row(
            "Histogram gradient boosting", "Boosting",
            lambda seed: ensemble.HistGradientBoostingRegressor(random_state=seed), sparse=False, nan=True,
            params=_HGB, epochs="hist_gradient_boosting"),
        "svm": _row("Support vector regression", "SVM", lambda seed: svm.SVR(), note=_SVM_NOTE, params=_SVM),
        "mlp": _row("Neural network (MLP)", "Neural network",
                    lambda seed: neural_network.MLPRegressor(random_state=seed),
                    note=_MLP_NOTE, params=_MLP, epochs="mlp"),
    }


def _tiny(clf):
    X = np.random.default_rng(0).normal(size=(30, 3))
    return X, (np.arange(30) % 2 if clf else X[:, 0])


def _tiny_fit(make, clf):
    try:
        make(0).fit(*_tiny(clf))
        return True
    except Exception:  # noqa: BLE001 -- a broken optional install just hides its row
        return False


@lru_cache
def _xgboost_gpu():
    """XGBoost silently falls back to the CPU when no GPU is visible (verified), so
    ask the fitted booster which device it really used."""
    import warnings
    import xgboost
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            booster = xgboost.XGBClassifier(device="cuda", n_estimators=1).fit(*_tiny(True)).get_booster()
        return json.loads(booster.save_config())["learner"]["generic_param"]["device"].startswith("cuda")
    except Exception:  # noqa: BLE001 -- no CUDA build / driver: no GPU for XGBoost
        return False


@lru_cache
def _catboost_gpu():
    import catboost
    from catboost.utils import get_gpu_device_count
    try:
        if get_gpu_device_count() < 1:
            return False
        catboost.CatBoostClassifier(task_type="GPU", iterations=1, verbose=0, allow_writing_files=False
                                    ).fit(*_tiny(True))
        return True
    except Exception:  # noqa: BLE001 -- GPU not usable by CatBoost: CPU
        return False


def _optional(task):
    clf = task == "classification"
    rows = {}
    try:
        import xgboost

        def xgb_defaults():  # get_params() is empty for XGBoost; read what a fit actually used
            booster = (xgboost.XGBClassifier if clf else xgboost.XGBRegressor)().fit(*_tiny(clf)).get_booster()
            tp = json.loads(booster.save_config())["learner"]["gradient_booster"]["tree_train_param"]
            return {"n_estimators": booster.num_boosted_rounds(), "learning_rate": round(float(tp["eta"]), 6),
                    "max_depth": int(tp["max_depth"])}

        xgb_gpu = _xgboost_gpu()
        rows["xgboost"] = _row(
            "XGBoost", "Boosting",
            lambda seed: (xgboost.XGBClassifier if clf else xgboost.XGBRegressor)(
                random_state=seed, **({"device": "cuda"} if xgb_gpu else {})), nan=True,
            params=_XGB, epochs="xgboost", resolve=lru_cache(xgb_defaults), device="gpu" if xgb_gpu else "cpu")
    except ImportError:
        pass
    try:
        import lightgbm
        rows["lightgbm"] = _row(
            "LightGBM", "Boosting",
            lambda seed: (lightgbm.LGBMClassifier if clf else lightgbm.LGBMRegressor)(random_state=seed, verbose=-1),
            nan=True, balanced=_CW if clf else None, params=_LGBM, epochs="lightgbm")
    except ImportError:
        pass
    try:
        import catboost

        def cat_defaults():  # get_params() is empty for CatBoost; learning rate depends on the data
            ap = (catboost.CatBoostClassifier if clf else catboost.CatBoostRegressor)(
                verbose=0, allow_writing_files=False).fit(*_tiny(clf)).get_all_params()
            return {"iterations": ap["iterations"], "depth": ap["depth"], "learning_rate": AUTO}

        cat_gpu = _catboost_gpu()
        rows["catboost"] = _row(
            "CatBoost", "Boosting",
            lambda seed: (catboost.CatBoostClassifier if clf else catboost.CatBoostRegressor)(
                random_seed=seed, verbose=0, allow_writing_files=False, **({"task_type": "GPU"} if cat_gpu else {})),
            nan=True, balanced={"auto_class_weights": "Balanced"} if clf else None,
            note="on GPU: curve appears after each fit and cancel waits for the fit (CatBoost GPU refuses callbacks)"
            if cat_gpu else "very slow on wide text data (measured 114 s on 1,000 SMS rows)",
            params=_CAT, epochs="catboost", resolve=lru_cache(cat_defaults), device="gpu" if cat_gpu else "cpu")
    except ImportError:
        pass
    return {k: r for k, r in rows.items() if _tiny_fit(r["make"], clf)}


# What each model is, in plain words -- rendered as-is by the UI (single source).
# Measured figures are from this machine (spec 2026-09-13/14).
ABOUT = {
    "baseline": {
        "what": "Always predicts the most common class (classification) or the training mean (regression).",
        "how": "Learns nothing from the features.",
        "good": "A reference point: every real model should beat it.",
        "watch": "Not a real model. A high baseline score just means one class dominates."},
    "logistic_regression": {
        "what": "A linear classifier that turns a weighted sum of the features into a probability.",
        "how": "Learns one weight per feature (per class) by minimising log loss, with L2 regularisation by default.",
        "good": "Many sparse features such as text; a fast, interpretable reference model.",
        "watch": "Only straight-line (linear) boundaries; unscaled numeric columns can slow convergence."},
    "linear_regression": {
        "what": "Fits a straight-line (weighted sum) relationship between the features and the target.",
        "how": "Ordinary least squares: picks the weights that minimise squared error.",
        "good": "Roughly linear relationships where you want readable coefficients.",
        "watch": "No regularisation: unstable with many or correlated features, and sensitive to outliers."},
    "ridge": {
        "what": "Linear regression with an L2 penalty that shrinks the weights.",
        "how": "Least squares plus alpha x the sum of squared weights.",
        "good": "Many correlated features, including text.",
        "watch": "Still linear; too large an alpha underfits."},
    "lasso": {
        "what": "Linear regression with an L1 penalty that can set weights exactly to zero.",
        "how": "Least squares plus alpha x the sum of absolute weights, solved by coordinate descent.",
        "good": "When only a few features are expected to matter.",
        "watch": "Among correlated features it keeps one fairly arbitrarily; may need more iterations to converge."},
    "elastic_net": {
        "what": "Linear regression that mixes the L1 (Lasso) and L2 (Ridge) penalties.",
        "how": "Least squares plus a blend of both penalties; l1_ratio sets the mix.",
        "good": "Many correlated features where some weights should still reach zero.",
        "watch": "Two settings to balance: alpha and l1_ratio."},
    "gaussian_nb": {
        "what": "Naive Bayes for continuous numbers.",
        "how": "Assumes each feature is normally distributed within each class and independent of the others.",
        "good": "Small, dense numeric datasets; trains almost instantly.",
        "watch": "Needs dense data (disabled when there are text columns); probabilities are often poorly calibrated."},
    "multinomial_nb": {
        "what": "Naive Bayes for counts or frequencies, such as word counts.",
        "how": "Estimates how often each feature occurs in each class, with alpha smoothing for unseen features.",
        "good": "Text classification; very fast even with many features.",
        "watch": "Needs non-negative inputs; assumes features are independent given the class."},
    "complement_nb": {
        "what": "A Naive Bayes variant built for uneven class sizes in text.",
        "how": "Estimates each class's feature weights from all the other classes (its complement).",
        "good": "Text data where some classes are much rarer than others.",
        "watch": "Needs non-negative inputs; same independence assumption as Multinomial naive Bayes."},
    "knn": {
        "what": "Predicts from the most similar training rows.",
        "how": "Keeps the training rows; for a new row it finds the k nearest by distance and takes their "
               "majority class (or average value).",
        "good": "Small, low-dimensional numeric data with scaled columns.",
        "watch": "Slow to predict on large data; distances lose meaning with many features such as text."},
    "decision_tree": {
        "what": "A flowchart of yes/no questions about the features.",
        "how": "Repeatedly splits the rows on the feature and threshold that best separate the target.",
        "good": "Rules a person can read; mixed column types; missing values are handled.",
        "watch": "Overfits easily unless depth or leaf size is limited; small data changes can reshape the tree."},
    "random_forest": {
        "what": "Many decision trees that vote (classification) or average (regression).",
        "how": "Each tree trains on a bootstrap sample of rows and a random subset of features at each split.",
        "good": "A strong, low-effort default on most tables; handles missing values.",
        "watch": "Slower and larger than one tree (can be large on disk); harder to interpret."},
    "extra_trees": {
        "what": "Like a random forest, but with random split thresholds.",
        "how": "Draws split thresholds at random instead of searching for the best one, then averages many trees.",
        "good": "Faster to train than a random forest; sometimes generalises better on noisy data.",
        "watch": "Usually needs more trees for stable results; harder to interpret."},
    "adaboost": {
        "what": "Boosting that focuses on the rows earlier models got wrong.",
        "how": "Trains small models (one-split trees by default) one after another, increasing the weight of "
               "misclassified rows.",
        "good": "Clean data where simple models underfit.",
        "watch": "Sensitive to noisy labels and outliers; measured 3.9 s on 1,000 SMS rows (slower than LightGBM)."},
    "gradient_boosting": {
        "what": "Trees added one at a time, each correcting the ensemble's remaining error.",
        "how": "Each new tree fits the gradient of the loss of the current ensemble, scaled by the learning rate.",
        "good": "Accurate on small-to-medium tabular data.",
        "watch": "Slow on large or wide data (measured 10.9 s on 1,000 SMS rows); training-loss curve only, "
                 "no early stopping on held-back rows."},
    "hist_gradient_boosting": {
        "what": "A faster gradient boosting that bins feature values first.",
        "how": "Buckets each feature into at most 255 bins, then builds boosted trees on the bins; handles "
               "missing values natively.",
        "good": "Large dense tabular data (tens of thousands of rows and more).",
        "watch": "Needs dense data (disabled when there are text columns); its curve appears only after fitting."},
    "svm": {
        "what": "Finds the boundary with the widest margin between classes (for regression, a tube around the target).",
        "how": "Optimises the margin using only the rows closest to the boundary (support vectors); kernels allow "
               "curved boundaries.",
        "good": "Small-to-medium data with many features, e.g. text with a linear kernel.",
        "watch": "Fit time grew ~4x per doubling of rows (measured); no probabilities by default, so log loss "
                 "and multiclass AUC show n/a."},
    "mlp": {
        "what": "A feed-forward neural network (multi-layer perceptron).",
        "how": "Layers of units with learned weights, trained epoch by epoch by backpropagation with the Adam optimiser.",
        "good": "Non-linear patterns when there are enough rows.",
        "watch": "Sensitive to its settings and to unscaled numbers; slow and memory-heavy on wide text "
                 "(first layer = features x units)."},
    "xgboost": {
        "what": "An optimised gradient-boosted tree library.",
        "how": "Adds regularised trees round by round; handles sparse inputs and missing values natively.",
        "good": "Strong accuracy on tabular data, with early stopping.",
        "watch": "Rounds, learning rate and depth interact; can overfit without early stopping."},
    "lightgbm": {
        "what": "A fast gradient-boosted tree library.",
        "how": "Grows trees leaf by leaf on histogram-binned features; handles sparse inputs and missing values.",
        "good": "Large or wide data where speed matters, with early stopping.",
        "watch": "Leaf-wise growth can overfit small datasets; limit the leaves per tree."},
    "catboost": {
        "what": "A gradient-boosted tree library that builds symmetric (oblivious) trees.",
        "how": "Adds depth-balanced trees round by round; defaults to 1,000 rounds with a data-dependent learning rate.",
        "good": "Tabular data with complex interactions, with early stopping.",
        "watch": "Its native category handling is not used here (preprocessing already encodes categories); "
                 "very slow on wide text on the CPU (measured 114 s on 1,000 SMS rows; 2.6x faster on the GPU). "
                 "On a GPU the curve appears only after each fit and a cancel waits for the fit to end."},
}

# One line per parameter: what changing it does. (model key, name) overrides name.
HELP = {
    "C": "Regularisation, inverted: smaller = simpler model, larger = follows the training data more closely.",
    "max_iter": "Most solver iterations; raise it if training stops before converging.",
    "var_smoothing": "Tiny value added to every feature's variance for numerical stability.",
    "alpha": "Smoothing for features never seen with a class; 0 = none (can give zero probabilities).",
    "n_neighbors": "How many nearest rows vote: small follows noise, large gives smoother predictions.",
    "weights": "uniform = every neighbour counts equally; distance = closer neighbours count more.",
    "n_estimators": "Number of trees: more = steadier results but slower training and a larger model.",
    "max_depth": "Longest chain of questions per tree: lower = simpler trees and less overfitting.",
    "min_samples_leaf": "Fewest training rows allowed in a final leaf: higher = smoother, less overfitting.",
    "learning_rate": "Step size per round: lower is usually more accurate but needs more rounds.",
    "kernel": "Boundary shape: linear = straight; rbf, poly, sigmoid = curved.",
    "l1_ratio": "Penalty mix: 0 = Ridge-style (L2), 1 = Lasso-style (L1).",
    "hidden_layer_sizes": "Units per hidden layer; 64,32 = two layers of 64 and 32 units.",
    "learning_rate_init": "Adam step size: too high can diverge, too low learns slowly.",
    "batch_size": "Rows per weight update; auto = min(200, rows).",
    "num_leaves": "Most leaves per tree: the main complexity (and overfitting) control in LightGBM.",
    "iterations": "Boosting rounds = epochs on the live chart.",
    "depth": "Depth of each symmetric tree: deeper = more complex interactions.",
}
HELP_BY_MODEL = {
    ("adaboost", "n_estimators"): "Boosting rounds: more rounds fit the training data more closely.",
    ("gradient_boosting", "n_estimators"): "Boosting stages = epochs on the live chart: more fits training data more closely.",
    ("gradient_boosting", "max_depth"): "Depth of each small tree added per stage.",
    ("hist_gradient_boosting", "max_iter"): "Boosting rounds = epochs on the live chart.",
    ("xgboost", "n_estimators"): "Boosting rounds = epochs on the live chart.",
    ("lightgbm", "n_estimators"): "Boosting rounds = epochs on the live chart.",
    ("mlp", "max_iter"): "Epochs: full passes over the training rows.",
    ("mlp", "alpha"): "L2 weight penalty: larger = simpler network, less overfitting.",
    ("ridge", "alpha"): "Regularisation strength: larger shrinks the weights more.",
    ("lasso", "alpha"): "Regularisation strength: larger sets more weights exactly to zero.",
    ("elastic_net", "alpha"): "Overall regularisation strength.",
}


def _describe(key, row):
    """Attach the plain-language description and one help line per parameter."""
    row["about"] = ABOUT[key]
    row["params"] = [{**p, "help": p["help"] or HELP_BY_MODEL.get((key, p["name"])) or HELP[p["name"]]}
                     for p in row["params"]]
    return row


@lru_cache
def table(task):
    rows = _classifiers() if task == "classification" else _regressors()
    rows.update(_optional(task))
    return {key: _describe(key, row) for key, row in rows.items()}


def incompatible(row, facts):
    """Why this model can't train on this data, or None. `facts` are measured by
    fitting the real preprocessing on the training split (train.facts)."""
    if facts["sparse"] and not row["sparse"]:
        return "needs dense data - your data has text columns"
    if facts["negative"] and row["nonneg"]:
        return "needs non-negative inputs - preprocessing produced negative values"
    if facts["nan"] and not row["nan"]:
        return "can't handle missing values - add an impute step on the Preprocessing tab"
    return None


def suggest(task, facts, rows):
    """Pre-ticked starting set with the reason for each (spec §5). A suggestion is
    never 'best' -- only CV results are. SVM is never pre-ticked (cost)."""
    def ok(key):
        return key in rows and incompatible(rows[key], facts) is None

    out = {}
    linear = "logistic_regression" if task == "classification" else "ridge"
    if ok(linear):
        out[linear] = "fast linear reference model"
    if ok("random_forest"):
        out["random_forest"] = "robust default on most tables, few settings to get wrong"
    for booster in ("lightgbm", "hist_gradient_boosting", "gradient_boosting"):
        if ok(booster):
            out[booster] = "gradient boosting, usually strongest on tabular data"
            break
    if task == "classification" and ok("multinomial_nb"):
        out["multinomial_nb"] = "all inputs are non-negative - fast and strong on word counts"
    return out


if __name__ == "__main__":
    import warnings
    from scipy.sparse import csr_matrix, hstack, random as sprand

    from training import params as P
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    n = 60
    dense = rng.normal(size=(n, 4))
    wide = sprand(n, 50, density=0.1, format="csr", random_state=0)
    with_nan = dense.copy()
    with_nan[::7, 0] = np.nan
    shapes = {  # data shape -> (matrix, the facts train.facts would report for it)
        "sparse, negatives": (hstack([csr_matrix(dense), wide]).tocsr(), {"sparse": True, "negative": True, "nan": False}),
        "sparse, non-negative": (hstack([csr_matrix(np.abs(dense)), wide]).tocsr(),
                                 {"sparse": True, "negative": False, "nan": False}),
        "dense, missing values": (with_nan, {"sparse": False, "negative": True, "nan": True}),
        "dense, non-negative": (np.abs(dense), {"sparse": False, "negative": False, "nan": False}),
    }
    epoch_adapters = {"mlp", "xgboost", "lightgbm", "catboost", "gradient_boosting", "hist_gradient_boosting"}
    for task, y in (("classification", np.arange(n) % 2), ("regression", rng.normal(size=n))):
        rows = table(task)
        assert {"baseline", "mlp", "xgboost", "lightgbm", "catboost"} <= set(rows), sorted(rows)
        for key, row in rows.items():
            for shape, (X, facts) in shapes.items():
                reason = incompatible(row, facts)
                try:
                    row["make"](0).fit(X, y)
                    trained = True
                except Exception:  # noqa: BLE001
                    trained = False
                assert trained == (reason is None), (task, key, shape, reason, trained)
            # defaults are real library values, JSON-safe, and rebuild a working model
            d = P.defaults(row)
            json.dumps(d)
            assert all(v is not None or p["none"] for p, v in zip(row["params"], d.values())), (task, key, d)
            kwargs, fit, _ = P.resolve(row, {})
            row["make"](0).set_params(**kwargs).fit(np.abs(dense), y)
            assert row["epochs"] in epoch_adapters | {None}, (key, row["epochs"])
            assert all(row["about"][f] for f in ("what", "how", "good", "watch")), (task, key)
            assert all(p["help"] for p in row["params"]), (task, key, [p["name"] for p in row["params"] if not p["help"]])
            assert bool(fit) == (row["epochs"] not in (None, "gradient_boosting")), (key, fit)
        picks = suggest(task, {"sparse": True, "negative": True, "nan": False}, rows)
        assert "random_forest" in picks and "lightgbm" in picks and "svm" not in picks, picks
    assert "multinomial_nb" in suggest("classification", {"sparse": True, "negative": False, "nan": False},
                                       table("classification"))
    # device: GPU rows really build GPU models; with the GPU hidden every row is CPU
    import os
    import subprocess
    import sys
    rows = table("classification")
    assert {k for k, r in rows.items() if r["device"] == "gpu"} <= {"xgboost", "catboost"}, rows
    assert (rows["xgboost"]["make"](0).get_params().get("device") == "cuda") == (rows["xgboost"]["device"] == "gpu")
    assert (rows["catboost"]["make"](0).get_params().get("task_type") == "GPU") == (rows["catboost"]["device"] == "gpu")
    hidden = subprocess.run(
        [sys.executable, "-c", "from training import models; print(sorted({r['device'] for r in "
                               "models.table('classification').values()}))"],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}, capture_output=True, text=True, check=True)
    assert hidden.stdout.strip().splitlines()[-1] == "['cpu']", hidden.stdout
    print("devices:", {k: r["device"] for k, r in rows.items() if r["device"] == "gpu"} or "all CPU (no usable GPU)")
    xgb_d, cat_d = P.defaults(table("classification")["xgboost"]), P.defaults(table("classification")["catboost"])
    assert xgb_d["n_estimators"] == 100 and xgb_d["learning_rate"] == 0.3 and xgb_d["max_depth"] == 6, xgb_d
    assert cat_d["iterations"] == 1000 and cat_d["depth"] == 6 and cat_d["learning_rate"] == AUTO, cat_d
    print(f"models self-check passed ({len(table('classification'))} classifiers, "
          f"{len(table('regression'))} regressors)")
