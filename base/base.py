from sklearn.base import BaseEstimator, TransformerMixin

class ColumnSelector(BaseEstimator, TransformerMixin):

        def __init__(self, cols: list):
            """ Custom sklearn transformer to select a set of columns.

            Attributes:
                cols (list of str) representing the columns to be selected 
                in a pandas DataFrame.

            """
            self.__cols = cols
            self.__pd_df = pd.DataFrame

        @property
        def cols(self):
            return self.__cols

        def get_feature_names(self):
            return self.__cols

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            assert isinstance(X, self.__pd_df), "`X` should be a pandas dataframe"
            return X.loc[:, self.__cols]


class NumericImputer(BaseEstimator, TransformerMixin):

    def __init__(self, method: str = "mean", fill_value=None):
        """ Custom sklearn transformer to impute numeric data when it is missing.

        Attributes:
            method (str) representing the method (mean/median/constant)
            fill_value (int/float) representing the constant value to be imputed 

        """
        assert method in ["mean", "median", "constant"], \
               "Allowed methods are `mean`, `median`, `constant`"
        if method == "constant":
            assert fill_value is not None, "Fill value must be provided for `constant`"
        self.__method = method
        self.__fill_value = fill_value
        self.__learned_values = {}
        self.__cols = []
        self.__pd_df = pd.DataFrame
        self.__np_mean = np.mean
        self.__np_median = np.median

    @property
    def method(self):
        return self.__method

    @property
    def fill_value(self):
        return self.__fill_value

    @property
    def learned_values(self):
        return self.__learned_values

    def __define_func(self):
        if self.__method == "mean":
            return self.__np_mean
        elif self.__method == "median":
            return self.__np_median

    def get_feature_names(self):
        return self.__cols

    def fit(self, X, y=None):
        assert isinstance(X, self.__pd_df), "`X` should be a pandas dataframe"
        X_ = X.copy()
        self.__cols = X_.columns
        if self.__method in ["mean", "median"]:
            func = self.__define_func()
            for column in X_.columns:
                self.__learned_values[column] = func(X_.loc[~X_[column].isnull(), column])
        elif self.__method == "constant":
            for column in X_.columns:
                self.__learned_values[column] = self.__fill_value
        return self

    def transform(self, X):
        assert isinstance(X, self.__pd_df), "`X` should be a pandas dataframe"
        X_ = X.copy()
        for column in X_.columns:
            X_.loc[X_[column].isnull(), column] = self.__learned_values[column]
        return X_


class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, method: str = "most_frequent", fill_value=None):
        """ Custom sklearn transformer to impute categorical data when it is missing.

        Attributes:
            method (str) representing the method (most_frequent/constant)
            fill_value (int/str) representing the constant value to be imputed 

        """
        assert method in ["most_frequent", "constant"], \
               "Allowed methods are `most_frequent`, `constant`"
        if method == "constant":
            assert fill_value is not None, "Fill value must be provided for `constant`"
        self.__method = method
        self.__fill_value = fill_value
        self.__learned_values = {}
        self.__cols = []
        self.__pd_df = pd.DataFrame

    @property
    def method(self):
        return self.__method

    @property
    def fill_value(self):
        return self.__fill_value

    @property
    def learned_values(self):
        return self.__learned_values

    def get_feature_names(self):
        return self.__cols

    def fit(self, X: pd.DataFrame, y=None):
        assert isinstance(X, self.__pd_df), "`X` should be a pandas dataframe"
        X_ = X.copy()
        self.__cols = X_.columns
        if self.__method == "most_frequent":
            for column in X_.columns:
                self.__learned_values[column] = X_.loc[:, column].value_counts(ascending=False).index[0]
        elif self.__method == "constant":
            for column in X_.columns:
                self.__learned_values[column] = self.__fill_value
        return self

    def transform(self, X):
        assert isinstance(X, self.__pd_df), "`X` should be a pandas dataframe"
        X_ = X.copy()
        for column in X_.columns:
            X_.loc[X_[column].isnull(), column] = self.__learned_values[column]
        return X_