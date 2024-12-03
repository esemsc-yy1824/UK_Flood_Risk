import os

from collections.abc import Sequence
from typing import List
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from scipy.stats import randint, uniform

from sklearn import set_config
import pyproj

set_config(transform_output="pandas")

from .model import *

__all__ = [
    "Tool",
    "_data_dir",
    "_example_dir",
    "flood_class_from_postcode_methods",
    "flood_class_from_location_methods",
    "house_price_methods",
    "local_authority_methods",
    "historic_flooding_methods",
]

_data_dir = os.path.dirname(__file__)

flood_class_from_postcode_methods = {
    "flood_risk_classifier": "flood_risk_classifier",
}
flood_class_from_location_methods = {
    "flood_risk_classifier": "flood_risk_classifier",
}
historic_flooding_methods = {
    "historic_flooding_classifier": "historic_flooding_classifier",
}
house_price_methods = {
    "house_regressor": "house_regressor",
}
local_authority_methods = {
    "local_authority_classifier": "local_authority_classifier",
}

IMPUTATION_CONSTANTS = {
    "soilType": "Unsurveyed/Urban",
    "elevation": 60.0,
    "distanceToWatercourse": 80,
    "localAuthority": np.nan,
}


class Tool(object):
    """Class to interact with a postcode database file."""

    def __init__(
        self,
        labelled_unit_data: str = "",
        unlabelled_unit_data: str = "",
        district_data: str = "",
    ):
        """
        Parameters
        ----------

        unlabelled_unit_data : str, optional
            Filename of a .csv file containing geographic location
            data for postcodes.

        labelled_unit_data: str, optional
            Filename of a .csv containing class labels for specific
            postcodes.

        sector_data : str, optional
            Filename of a .csv file containing information on households
            by postcode sector.

        district_data : str, optional
            Filename of a .csv file containing information on households
            by postcode district.

        additional_data: dict, optional
            Dictionary containing additional .csv files containing addtional
            information on households.
        """

        # Set defaults if no inputs provided
        if labelled_unit_data == "":
            labelled_unit_data = os.path.join(_data_dir, "../resources_data/postcodes_labelled.csv")

        if unlabelled_unit_data == "":
            unlabelled_unit_data = os.path.join(
                _data_dir, "../example_data/postcodes_unlabelled.csv"
            )

        if district_data == "":
            district_data = os.path.join(_data_dir, "../resources_data/district_data.csv")

        # Load the data and preprocess
        self._postcodedb = pd.read_csv(labelled_unit_data).drop_duplicates()
        self._postcodedb["postcode"] = self._postcodedb["postcode"].apply(
            self.standardise_UK_postcode
        )
        self._postcodedb = self.impute_missing_values(self._postcodedb)

        self._unlabelled_postcodes = pd.read_csv(unlabelled_unit_data).drop_duplicates()
        self._unlabelled_postcodes["postcode"] = self._unlabelled_postcodes["postcode"].apply(
            self.standardise_UK_postcode
        )

        self.models = Models()
        
        self.trained_models = {}

        self.le = LabelEncoder()

    def fit(
        self,
        models: List[str] = [],
        update_labels: str = "",
        update_hyperparameters: bool = False,
    ):
        """Fit/train models using a labelled set of samples.

        Parameters
        ----------

        models : sequence of model keys
            Models to fit/train
        update_labels : str, optional
            Filename of a .csv file containing an updated
            labelled set of samples
            in the same format as the original labelled set.

            If not provided, the data set provided at
            initialisation is used.
        update_hyperparameters : bool, optional
            If True, models may tune their hyperparameters, where
            possible. If False, models will use their default hyperparameters.
        Examples
        --------
        >>> tool = Tool()
        >>> fcp_methods = list(flood_class_from_postcode_methods.keys())
        >>> tool.fit(fcp_methods, update_labels='new_labels.csv')  # doctest: +SKIP
        >>> classes = tool.predict_flood_class_from_postcode(
        ...    ['M34 7QL'], fcp_methods[0])  # doctest: +SKIP
        """

        X_train = self._postcodedb.drop(
            columns=["postcode", "riskLabel", "historicallyFlooded", "medianPrice"]
        )

        if update_labels:
            print("updating labelled sample file")
            self._postcodedb = pd.read_csv(update_labels)

        for model_key in models:
            if model_key in flood_class_from_postcode_methods:
                self.trained_models[model_key] = self.models.flood_risk_model()

                if update_hyperparameters:
                    param_grid = {
                        "classifier__n_estimators": randint(50, 200),
                        "classifier__max_depth": randint(3, 10),
                        "classifier__learning_rate": uniform(0.01, 0.2),
                    }

                    random_search = RandomizedSearchCV(
                        estimator=self.trained_models[model_key],
                        param_distributions=param_grid,
                        n_iter=10,
                        scoring="accuracy",
                        cv=5,
                        verbose=0,
                        n_jobs=-1,
                        random_state=42,
                    )

                    random_search.fit(
                        X_train, self.le.fit_transform(self._postcodedb["riskLabel"])
                    )
                    self.trained_models[model_key] = random_search.best_estimator_
                else:
                    self.trained_models[model_key].fit(
                        X_train, self.le.fit_transform(self._postcodedb["riskLabel"])
                    )

            elif model_key in flood_class_from_location_methods:
                self.trained_models[model_key] = self.models.flood_risk_model()

                if update_hyperparameters:
                    param_grid = {
                        "classifier__n_estimators": randint(50, 200),
                        "classifier__max_depth": randint(3, 10),
                        "classifier__learning_rate": uniform(0.01, 0.2),
                    }

                    random_search = RandomizedSearchCV(
                        estimator=self.trained_models[model_key],
                        param_distributions=param_grid,
                        n_iter=10,
                        scoring="accuracy",
                        cv=5,
                        verbose=0,
                        n_jobs=-1,
                        random_state=42,
                    )

                    random_search.fit(
                        X_train, self.le.fit_transform(self._postcodedb["riskLabel"])
                    )
                    self.trained_models[model_key] = random_search.best_estimator_
                else:
                    self.trained_models[model_key].fit(
                        X_train, self.le.fit_transform(self._postcodedb["riskLabel"])
                    )

            elif model_key in historic_flooding_methods:
                self.trained_models[model_key] = self.models.historic_flooding_model()

                if update_hyperparameters:
                    param_grid = {
                        "classifier__n_estimators": randint(50, 200),
                        "classifier__max_depth": randint(3, 10),
                        "classifier__learning_rate": uniform(0.01, 0.2),
                    }

                    random_search = RandomizedSearchCV(
                        estimator=self.trained_models[model_key],
                        param_distributions=param_grid,
                        n_iter=10,
                        scoring="accuracy",
                        cv=5,
                        verbose=0,
                        n_jobs=-1,
                        random_state=42,
                    )

                    random_search.fit(X_train, self._postcodedb["historicallyFlooded"])
                    self.trained_models[model_key] = random_search.best_estimator_
                else:
                    self.trained_models[model_key].fit(
                        X_train, self._postcodedb["historicallyFlooded"]
                    )

            elif model_key in house_price_methods:
                self.trained_models[model_key] = self.models.house_price_model()

                if update_hyperparameters:
                    param_grid = {
                        "regressor__n_estimators": [100, 200, 300],
                        "regressor__max_depth": [3, 5, 10],
                        "regressor__learning_rate": [0.01, 0.1],
                        "regressor__subsample": [0.8, 1.0],
                        "regressor__colsample_bytree": [0.8, 1.0],
                        "regressor__reg_alpha": [0, 0.1],
                        "regressor__reg_lambda": [1, 2],
                    }

                    random_search = RandomizedSearchCV(
                        estimator=self.trained_models[model_key],
                        param_distributions=param_grid,
                        n_iter=10,
                        cv=5,
                        scoring="neg_root_mean_squared_error",
                        random_state=42,
                        verbose=0,
                        n_jobs=-1,
                    )

                    random_search.fit(X_train, self._postcodedb["medianPrice"])
                    self.trained_models[model_key] = random_search.best_estimator_
                else:
                    self.trained_models[model_key].fit(
                        X_train, self._postcodedb["medianPrice"]
                    )

            elif model_key in local_authority_methods:
                self.trained_models[model_key] = (
                    self.models.local_authority_modelX_train()
                )

                if update_hyperparameters:
                    param_grid = {
                        "classifier__n_estimators": randint(50, 200),
                        "classifier__max_depth": randint(3, 10),
                        "classifier__learning_rate": uniform(0.01, 0.2),
                    }

                    random_search = RandomizedSearchCV(
                        estimator=self.trained_models[model_key],
                        param_distributions=param_grid,
                        n_iter=10,
                        scoring="accuracy",
                        cv=5,
                        verbose=0,
                        n_jobs=-1,
                        random_state=42,
                    )

                    random_search.fit(X_train, self._postcodedb["localAuthority"])
                    self.trained_models[model_key] = random_search.best_estimator_
                else:
                    self.trained_models[model_key].fit(
                        X_train, self._postcodedb["localAuthority"]
                    )
            else:
                raise ValueError(f"Unknown model key: {model_key}")

    def impute_missing_values(
        self,
        data: pd.DataFrame,
        method: str = "knn",
        n_neighbors: int = 4,
        constant_values: dict = IMPUTATION_CONSTANTS,
    ) -> pd.DataFrame:
        """Impute missing values in a dataframe.

        Parameters
        ----------

        dataframe : pandas.DataFrame
            DataFrame (in the format of the unlabelled postcode data)
            potentially containing missing values as NaNs, or with missing
            columns.

        method : str, optional
            Method to use for imputation. Options include:
            - 'mean', to use the mean for the labelled dataset
            - 'constant', to use a constant value for imputation
            - 'knn' to use k-nearest neighbours imputation from the
              labelled dataset

        constant_values : dict, optional
            Dictionary containing constant values to
            use for imputation in the format {column_name: value}.
            Only used if method is 'constant'.

        Returns
        -------

        pandas.DataFrame
            DataFrame with missing values imputed.

        Examples
        --------

        >>> tool = Tool()
        >>> missing = os.path.join(_example_dir, 'postcodes_missing_data.csv')
        >>> data = pd.read_csv(missing)
        >>> data = tool.impute_missing_values(data)  # doctest: +SKIP
        """
        if method not in ["mean", "constant", "knn"]:
            raise ValueError(
                f"Unsupported method '{method}'. Choose from 'mean', 'constant', or 'knn'."
            )

        df = data.copy()
        postcode = df["postcode"]
        df_drop_postcode = df.drop(columns=["postcode"])

        if df_drop_postcode.isnull().all().any():  # if any column in a DataFrame contains only NaN

            raise ValueError(
                    "DataFrame contains only NaN values. Imputation cannot be performed."
            )

        if method == "mean":
            numeric_col = df.select_dtypes(include=[np.number]).columns
            df_drop_postcode[numeric_col] = df_drop_postcode[numeric_col].fillna(
                df_drop_postcode[numeric_col].mean(numeric_only=True)
            )
        elif method == "constant":
            if constant_values is None:
                raise ValueError(
                    "Constant values must be provided for 'constant' imputation."
                )
            for col, value in constant_values.items():
                if col in df_drop_postcode.columns:
                    df_drop_postcode[col] = df_drop_postcode[col].fillna(value)
        elif method == "knn":
            cat_col = df_drop_postcode.select_dtypes(include=[object]).columns
            le = OrdinalEncoder()
            df_drop_postcode[cat_col] = le.fit_transform(df_drop_postcode[cat_col])
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df_drop_postcode = imputer.fit_transform(df_drop_postcode)
            # Decode back to original categories
            imputed_decoded = le.inverse_transform(df_drop_postcode[cat_col])
            df_drop_postcode[cat_col] = imputed_decoded

        return pd.concat([postcode, df_drop_postcode], axis=1)

    def predict_flood_class_from_postcode(
        self, postcodes: Sequence[str], method: str = "flood_risk_classifier"
    ) -> pd.Series:
        """
        Generate series predicting flood probability classification
        for a collection of poscodes.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcode units.
        method : str (optional)
            optionally specify (via a key in the
            `get_flood_class_from_postcode_methods` dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by postcodes.
            Returns NaN for postcode units not in the available postcode files.
        """
        postcodes = [self.standardise_UK_postcode(pc) for pc in postcodes]

        if method == "all_zero_risk":
            return pd.Series(
                data=np.ones(len(postcodes), int),
                index=np.asarray(postcodes),
                name="riskLabel",
            )

        elif method == "flood_risk_classifier":
            df_postcodes = self._unlabelled_postcodes[
                self._unlabelled_postcodes["postcode"].isin(postcodes)
            ]

            # Impute missing values
            df_postcodes = self.impute_missing_values(df_postcodes)

            self.fit([method], update_hyperparameters=True)

            y_pred = self.trained_models[method].predict(
                df_postcodes.drop(columns=["postcode"])
            )

            y_pred = self.le.inverse_transform(y_pred)

            return pd.Series(
                data=y_pred,
                index=np.asarray(postcodes)[
                    np.isin(
                        np.asarray(postcodes), self._unlabelled_postcodes["postcode"]
                    )
                ],
                name="riskLabel",
                dtype="int64",
            )

        else:
            raise NotImplementedError(f"method {method} not implemented")

    def predict_historic_flooding(
        self, postcodes: Sequence[str], method: str = "historic_flooding_classifier"
    ) -> pd.Series:
        """
        Generate series predicting whether a collection of postcodes
        has experienced historic flooding.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcode units.
        method : str (optional)
            optionally specify (via a key in the
            historic_flooding_methods dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series indicating whether a postcode experienced historic
            flooding, indexed by the postcodes.
        """
        postcodes = [self.standardise_UK_postcode(pc) for pc in postcodes]

        if method == "all_false":
            return pd.Series(
                data=np.full(len(postcodes), False),
                index=np.asarray(postcodes),
                name="historicallyFlooded",
            )

        elif method == "historic_flooding_classifier":
            df_postcodes = self._unlabelled_postcodes[
                self._unlabelled_postcodes["postcode"].isin(postcodes)
            ]

            # Impute missing values
            df_postcodes = self.impute_missing_values(df_postcodes)

            self.fit([method], update_hyperparameters=True)

            y_pred = self.trained_models[method].predict(
                df_postcodes.drop(columns=["postcode"])
            )

            y_pred = y_pred.astype(np.int64)

            return pd.Series(
                data=y_pred,
                index=np.asarray(postcodes)[
                    np.isin(
                        np.asarray(postcodes), self._unlabelled_postcodes["postcode"]
                    )
                ],
                name="historicallyFlooded",
            )

        else:
            raise NotImplementedError(f"method {method} not implemented")

    def estimate_annual_flood_economic_risk(
        self, postcodes: Sequence[str], risk_labels: Union[pd.Series, None] = None
    ) -> pd.Series:
        """
        Return a series of estimates of the total economic property risk
        for a collection of postcodes.

        Risk is defined here as a damage coefficient multiplied by the
        value under threat multiplied by the probability of an event.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcode units.
        risk_labels: pandas.Series (optional)
            optionally provide a Pandas Series containing flood risk
            classifiers, as predicted by get_flood_class_from_postcodes.

        Returns
        -------

        pandas.Series
            Series of total annual economic flood risk estimates indexed
            by postcode.
        """
        postcodes = [self.standardise_UK_postcode(pc) for pc in postcodes]

        risk_labels = (
            risk_labels or self.predict_flood_class_from_postcode(postcodes).values
        )

        if len(risk_labels) != len(postcodes):
            raise ValueError("risk_labels must be the same length as postcodes")

        risk_to_probability = {
            7: 0.05,
            6: 0.03,
            5: 0.02,
            4: 0.01,
            3: 0.005,
            2: 0.002,
            1: 0.001,
        }

        total_values = self.estimate_total_value(postcodes)

        return pd.Series(
            data=[
                0.5 * value * risk_to_probability[risk_label]
                for value, risk_label in zip(total_values.values, risk_labels)
            ],
            index=np.asarray(postcodes),
            name="economicRisk",
        )

    def standardise_UK_postcode(self, postcode: str, is_sector=False) -> str:
        """
        Standardise a postcode to upper case and ensure it has a space in the middle.

        This will also work for sectors

        Parameters
        ----------

        postcode : str
            Postcode to standardise.

        Returns
        -------

        str
            Standardised postcode.
        """
        postcode = postcode.replace(" ", "").upper()
        if len(postcode) > 3:
            return postcode[:-3] + " " + postcode[-3:]
        return postcode

    def make_output(
        self,
        postcodes=None,
        eastings=None,
        northings=None,
        longitudes=None,
        latitudes=None,
        path="output/output.csv",
    ):
        """
        Concatenate the outputs of all predict_* methods.

        Parameters
        ----------
        postcodes : sequence of strs, optional
            Sequence of postcode units.
        eastings : sequence of floats, optional
            Sequence of OSGB36 eastings.
        northings : sequence of floats, optional
            Sequence of OSGB36 northings.
        longitudes : sequence of floats, optional
            Sequence of WGS84 longitudes.
        latitudes : sequence of floats, optional
            Sequence of WGS84 latitudes.

        Returns
        -------
        pandas.DataFrame
            Combined output from all prediction functions.
        """
        output = pd.DataFrame()

        if postcodes is not None:
            flood_risk_postcode = self.predict_flood_class_from_postcode(
                postcodes
            ).to_frame()
            house_prices = self.predict_median_house_price(postcodes).to_frame()
            historical_flood = self.predict_historic_flooding(postcodes).to_frame()
            output = pd.concat(
                [flood_risk_postcode, house_prices, historical_flood], axis=1
            )

        if eastings is not None and northings is not None:
            flood_risk_osgb36 = self.predict_flood_class_from_OSGB36_location(
                eastings, northings
            ).to_frame()
            local_authority = self.predict_local_authority(
                eastings, northings
            ).to_frame()
            output = pd.concat([flood_risk_osgb36, local_authority], axis=1)

        if longitudes is not None and latitudes is not None:
            flood_risk_wgs84 = self.predict_flood_class_from_WGS84_locations(
                longitudes, latitudes
            )
            output = flood_risk_wgs84.to_frame()

        output.to_csv(path)
    
    def drop_noRecord_cat_row(self, raw_df):
        """
        Drop rows with unrecognized categorical values in specific columns.

        This method filters rows in the input DataFrame to include only those 
        with categorical values present in a predefined list. The lists of valid 
        categorical values are loaded from an external CSV file.

        Parameters
        ----------
        raw_df : pd.DataFrame
            The input DataFrame containing raw data with categorical columns.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing rows where the `soilType` and `localAuthority` 
            columns have values recognized in the predefined lists.

        Notes
        -----
        The method uses a resource file `postcodes_labelled.csv` located in the 
        `./resources_data/` directory to determine valid categorical values.

        Examples
        --------
        >>> raw_df = pd.DataFrame({
        ...     'soilType': ['Clay', 'Silt', 'Unknown'],
        ...     'localAuthority': ['CouncilA', 'CouncilB', 'Invalid']
        ... })
        >>> models = Models()
        >>> filtered_df = models.drop_noRecord_cat_row(raw_df)
        >>> filtered_df
        soilType localAuthority
        0      Clay       CouncilA
        1      Silt       CouncilB
        """
        cat_recorded = pd.read_csv("./resources_data/postcodes_labelled.csv")[['soilType', 'localAuthority']]
        
        soilType = [i for i in cat_recorded['soilType'].unique()]
        localAuthority = [i for i in cat_recorded['localAuthority'].unique()]

        for col in raw_df.columns:
            if col == 'soilType':
                raw_df_soilType = raw_df[raw_df[col].isin(soilType)]
            elif col == 'localAuthority':
                raw_df_localAuthority = raw_df[raw_df[col].isin(localAuthority)]
        
        df = pd.concat([raw_df_soilType, raw_df_localAuthority], axis=0)

        return df
    
    def easting_northing_to_lat_lon(self, df, easting_col, northing_col):
        """
        Convert easting and northing coordinates to latitude and longitude.

        This method transforms coordinates from the British National Grid (OSGB36) 
        to the WGS84 geographic coordinate system (latitude and longitude) and 
        appends the results as new columns in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing easting and northing columns.
        easting_col : str
            The name of the column in the DataFrame containing easting values.
        northing_col : str
            The name of the column in the DataFrame containing northing values.

        Returns
        -------
        pd.DataFrame
            The original DataFrame with two new columns: `Latitude` and `Longitude`, 
            containing the converted geographic coordinates.

        Notes
        -----
        - The transformation uses EPSG:27700 for the British National Grid (OSGB36) 
        and EPSG:4326 for WGS84.
        - Ensure the input data contains valid easting and northing values 
        to avoid transformation errors.

        Examples
        --------
        >>> df = pd.DataFrame({
        ...     'Easting': [400000, 500000],
        ...     'Northing': [300000, 200000]
        ... })
        >>> models = Models()
        >>> result_df = models.easting_northing_to_lat_lon(df, 'Easting', 'Northing')
        >>> result_df[['Latitude', 'Longitude']]
            Latitude  Longitude
        0  52.6576  -1.5184
        1  52.2053  -0.1218
        """
        
        latitudes = []
        longitudes = []
        
        # Define the projection for the British National Grid (OSGB36) and WGS84
        # EPSG:27700 is for OSGB36 (British National Grid)
        transformer = pyproj.Transformer.from_crs("epsg:27700", "epsg:4326", always_xy=True)
        
        for _, row in df.iterrows():
            easting = row[easting_col]
            northing = row[northing_col]
            
            lon, lat = transformer.transform(easting, northing)
            
            latitudes.append(lat)
            longitudes.append(lon)
        
        df['Latitude'] = latitudes
        df['Longitude'] = longitudes
        
        return df
