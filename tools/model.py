import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    RobustScaler,
    FunctionTransformer,
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
)

from sklearn import set_config

set_config(transform_output="pandas")



class Models:
    """
    A class to build machine learning pipelines for postcode-related predictions.

    This class provides methods to create machine learning models for tasks such as
    flood risk prediction, historic flooding classification, house price estimation,
    and local authority prediction. Each method includes preprocessing steps tailored
    to the specific task.

    Attributes
    ----------
    cat_pipeline : sklearn.pipeline.Pipeline
        A preprocessing pipeline for categorical data using `OrdinalEncoder`.
    """

    def __init__(self) -> None:
        """
        Initialize the Models class with a categorical preprocessing pipeline.
        """
        pass


    def flood_risk_model(self) -> str:
        """
        Create a pipeline to predict flood risk using a random forest classifier.

        This method preprocesses the input data by encoding categorical features and
        optionally dropping specified columns, then applies a Random Forest Classifier
        for predictions.

        Parameters
        ----------
        data : pd.DataFrame
            The input data to train the model, containing both features and target.
        cols : list of str, optional
            Columns to drop from the input data (default is an empty list).

        Returns
        -------
        sklearn.pipeline.Pipeline
            A pipeline object that preprocesses the input data and predicts flood risk.

        Examples
        --------
        >>> models = Models()
        >>> data = pd.DataFrame({'feature1': ['A', 'B'], 'feature2': [1, 2]})
        >>> pipeline = models.flood_risk_model(data, cols=['feature2'])
        >>> type(pipeline)
        <class 'sklearn.pipeline.Pipeline'>
        """

        preprocessor = Pipeline(
            [
                (
                    "drop_cols",
                    FunctionTransformer(
                        lambda X: self._drop_columns(
                            X, ["postcode","easting", "northing"]
                        ),
                        validate=False,
                    ),
                ),
                (
                    "scale_numerical",
                    FunctionTransformer(
                        lambda X: self._scale_numerical(X, scaling_type="standard"),
                        validate=False,
                    ),
                ),
                (
                    "cat_encoding",
                    FunctionTransformer(self._encode_categoricals, validate=False),
                ),
            ]
        )

        pipeline = Pipeline(
            [("preprocessor", preprocessor), ("classifier", RandomForestClassifier(random_state=42))]
            # [("preprocessor", self.flood_risk_preprocessor), ("classifier", RandomForestClassifier(random_state=42))]
        )
        return pipeline

    def _drop_columns(self, data: pd.DataFrame, cols_to_drop: list) -> pd.DataFrame:
        """
        Drop specified columns from the given DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            The input DataFrame from which columns will be dropped.
        cols_to_drop : list of str
            List of column names to be dropped from the DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the specified columns removed.

        Examples
        --------
        >>> data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        >>> cols_to_drop = ['B', 'C']
        >>> result = self._drop_columns(data, cols_to_drop)
        >>> result
           A
        0  1
        1  2
        2  3
        """
        return data.drop(columns=cols_to_drop)
    

    def _scale_numerical(
        self, data: pd.DataFrame, scaling_type: str = "standard", excluded_cols: list = None
    ) -> pd.DataFrame:
        """
        Scale numerical columns in the given DataFrame using the specified scaling method.
        Parameters
        ----------
        data : pd.DataFrame
            The input DataFrame containing numerical columns to be scaled.
        scaling_type : str, optional
            The type of scaling to apply. Options are 'standard', 'minmax', or 'robust'.
            Default is 'standard'.
        Returns
        -------
        pd.DataFrame
            A DataFrame with the numerical columns scaled according to the specified method.
        Raises
        ------
        ValueError
            If an unsupported scaling_type is provided.
        Examples
        --------
        >>> data = pd.DataFrame({'A': [1.0, 2.0, 3.0], 'B': [4.0, 5.0, 6.0]})
        >>> result = self._scale_numerical(data, scaling_type='minmax')
        >>> result
             A    B
        0  0.0  0.0
        1  0.5  0.5
        2  1.0  1.0
        """

        if excluded_cols is not None:
            excluded_data = data[excluded_cols]
            data = data.drop(columns=excluded_cols, inplace=False)
        num_columns = self._get_numerical_columns(data)

        # Select the scaler based on scaling_type
        if scaling_type == "standard":
            scaler = StandardScaler()
        elif scaling_type == "minmax":
            scaler = MinMaxScaler()
        elif scaling_type == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(
                "Unsupported scaling_type. Use 'standard', 'minmax', or 'robust'."
            )

        # Apply scaling to numerical columns
        if not num_columns.empty:
            scaled_values = scaler.fit_transform(data[num_columns])
            scaled_df = pd.DataFrame(
                scaled_values, index=data.index, columns=num_columns
            )
            # Replace scaled columns in the original DataFrame
            data[num_columns] = scaled_df

        if excluded_cols is not None:
            return pd.concat([data, excluded_data], axis=1)
        return data


    def _encode_categoricals(self, data):
        """
        Encode categorical columns in the given DataFrame using one-hot encoding.
        Parameters
        ----------
        data : pd.DataFrame
            The input DataFrame containing categorical columns to be encoded.
        Returns
        -------
        pd.DataFrame
            A DataFrame with the categorical columns encoded using one-hot encoding.
            Original categorical columns are dropped and replaced with their encoded counterparts.
        Examples
        --------
        >>> data = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': [1, 2, 3]})
        >>> result = self._encode_categoricals(data)
        >>> result
           B  A_b
        0  1    0
        1  2    1
        2  3    0
        """

        cat_columns = self._get_categorical_columns(data)

        if cat_columns.empty:
            return data

        encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown='ignore')
        # encoder = OneHotEncoder(sparse_output=False, drop="first")
        encoded = encoder.fit_transform(data[cat_columns])

        encoded_df = pd.DataFrame(
            encoded,
            index=data.index,
            columns=encoder.get_feature_names_out(cat_columns),
        )

        data = data.drop(columns=cat_columns)
        data = pd.concat([data, encoded_df], axis=1)
        return data

    def _get_numerical_columns(self, data):
        numeric_col = data.select_dtypes(include=[np.number]).columns
        return numeric_col
    
    def _get_categorical_columns(self, data):
        cat_columns = data.select_dtypes(include=["object", "category"]).columns
        return cat_columns