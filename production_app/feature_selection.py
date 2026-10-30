import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from production_app.config import settings
from production_app.logging_config import get_logger
from production_app.preprocessing import build_preprocessor

logger = get_logger(__name__)


def select_features(
    X_train: pd.DataFrame, y_train: pd.Series, c: float = settings.selection_c
) -> list[str]:
    l1_model = LogisticRegression(penalty="l1", solver="liblinear", C=c)
    selector = Pipeline(
        [
            ("scaler", build_preprocessor()),
            ("select", SelectFromModel(l1_model)),
        ]
    )
    selector.fit(X_train, y_train)

    kept = selector["select"].get_support()
    selected = list(X_train.columns[kept])

    if not selected:
        raise ValueError(f"Feature selection kept no features (C={c}); use a larger C")

    logger.info(
        "Feature selection (L1, C=%s): kept %d of %d features: %s",
        c,
        len(selected),
        X_train.shape[1],
        selected,
    )
    return selected
