from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


"""

Pipeline (

              1. MinMaxScaler -> num
                                        -> 3. preprocessed_df
              2. LabelEncoder -> cat

)

"""

def preprocessing_target_pipeline(target_column):
    """
    Preprocessing pipeline for the target variable (failure).
    Imputes missing values with the most frequent category and encodes with OrdinalEncoder.
    """
    target_preprocessing = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values
            ('encode', OrdinalEncoder())
        ]
    )
    return target_preprocessing


def preprocessing_pipeline(numeric_features, categorical_features):
    """
    Preprocessing pipeline for numerical and categorical variables.
    Applies imputation and scaling to numerics, imputation and one-hot encoding to categoricals.
    """
    numeric_preprocessing = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values
            ('scaler', MinMaxScaler())
        ]
    )
    categorical_preprocessing = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values
            ('encode', OneHotEncoder())
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ('num', numeric_preprocessing, numeric_features),
            ('cat', categorical_preprocessing, categorical_features)
        ]
    )
    return preprocessor