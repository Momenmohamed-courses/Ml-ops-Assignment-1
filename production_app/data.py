import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from production_app.config import settings
from production_app.logging_config import get_logger

logger = get_logger(__name__)

def load_dataset()->tuple[pd.DataFrame,pd.Series] :
    data=load_breast_cancer(as_frame=True)
    X=data.data
    y=data.target
    logger.info("Dataset Loaded: %d rows , %d features ",X.shape[0],X.shape[1])
    return X,y

def split_data(X: pd.DataFrame,
               y: pd.Series,
               test_size: float = settings.test_size,
               random_seed: int = settings.random_seed) -> tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]:
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = test_size,random_state = random_seed,stratify = y)
    logger.info("Split data: %d train rows, %d test rows",len(X_train),len(X_test))
    return X_train,X_test,y_train,y_test