import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from production_app.config import settings
from production_app.logging_config import get_logger
from production_app.preprocessing import build_preprocessor

logger = get_logger(__name__)

def build_pipeline(max_iter: int = settings.max_iter) -> Pipeline:
    return Pipeline(
        [
            ("scaler", build_preprocessor()),
            ("model", LogisticRegression(max_iter=max_iter)),
        ]
    )
                    
def train_model(X_train: pd.DataFrame, y_train : pd.Series) -> Pipeline:
    logger.info("Training started on %d rows",len(X_train))
    pipeline=build_pipeline()
    pipeline.fit(X_train,y_train)
    logger.info("Training finished")
    return pipeline


