import subprocess
subprocess.call(['pip', 'install', 'lightgbm'])
subprocess.call(['pip', 'install', 'sagemaker'])
subprocess.call(['pip', 'install', 'ydata_profiling'])
subprocess.call(['pip', 'install', 'shap'])

import pandas as pd
import numpy as np
import math
import json
import sqlite3
import datetime
import os
import logging
import boto3
import sagemaker
import pickle
import tarfile
import joblib
import argparse
import matplotlib.pyplot as plt
import io
import dill

from ydata_profiling import ProfileReport
import shap
from pandas.plotting import table

from lightgbm import LGBMClassifier

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import (GridSearchCV, train_test_split)
from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder, StandardScaler)
from sklearn.pipeline import (FeatureUnion, Pipeline)
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, ConfusionMatrixDisplay, f1_score, jaccard_score, precision_score,\
                             precision_recall_curve, recall_score, roc_auc_score, roc_curve)

# Since using "ColumnsTransformers" the usage of the endpoint does not work because of the "feature_names_in_", it was needed to bring the ground classes that build it
# Due to this fact "BaseEstimator" and "TransformerMixin" were brought back to rebuild few functions. The same happened for SimpleImputer, note that every class brought
# has a method called "get_feature_names(self)". In local machine using "Columns Transformers" and "SimpleImputers" works well.

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

class StarbucksProject(ColumnSelector, NumericImputer, CategoricalImputer):


    def __init__(self):
        """
        While the initiation of the class, the data and connections are already done, name of the files and folders are not supposed to be changed.
        """
        self.portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
        self.profile = pd.read_json('data/profile.json', orient='records', lines=True)
        self.transcript = pd.read_json('data/transcript.json', orient='records', lines=True)
        self._conn = sqlite3.connect(':memory:')
        self._session = sagemaker.Session()
        self._s3 = boto3.client('s3')
        self.role = sagemaker.get_execution_role()
        self.bucket = self._session.default_bucket()
        self.region = self._session.boto_region_name
        self.framework_version = "0.23-1"
        self.training_image_uri = sagemaker.image_uris.retrieve(\
                                framework = "sklearn",\
                                region = self.region,\
                                version = self.framework_version,\
                                py_version="py3")
        self.thresh_params = {'bogo': 0.50,
                              'discount': 0.50,
                              'informational': 0.50}
        self._transcript_completed_query = '''
            WITH TMP_DATA AS (
                SELECT *,
                       ROW_NUMBER() OVER(
                           PARTITION BY gt.customer_id, gt.offer_id, gt.time_completion
                           ORDER BY gt.time_for_completion ASC
                           ) AS simultaneos_offers
                FROM (
                    SELECT doc.customer_id,
                           doc.offer_id,
                           doc.time AS time_completion,
                           dor.time AS time_receive,
                           (doc.time - dor.time) AS time_for_completion,
                           doc.simultaneos_completion,
                           ROW_NUMBER() OVER(
                               PARTITION BY dor.customer_id, dor.offer_id, dor.time
                               ORDER BY dor.time
                           ) AS offer_receive_stack
                    FROM (
                        SELECT *,
                               COUNT() OVER(
                                   PARTITION BY doc.customer_id, doc.offer_id, doc.time
                                   ) AS simultaneos_completion
                        FROM doc
                    ) doc
                    LEFT JOIN dor
                    ON doc.customer_id = dor.customer_id
                    AND doc.offer_id = dor.offer_id
                    AND doc.time >= dor.time
                    AND doc.time <= dor.time_limit
                ) gt
            ),
            SIMULTANEOS AS (
                SELECT s.customer_id, s.offer_id, s.time_completion, s.time_receive, s.time_for_completion
                FROM (
                    SELECT *,
                           ROW_NUMBER() OVER(
                               PARTITION BY s.customer_id, s.offer_id, s.simultaneos_completion
                               ORDER BY s.time_receive ASC
                                       ) AS amount_combinations_found,
                           MAX(s.simultaneos_completion) OVER(
                               PARTITION BY s.customer_id, s.offer_id, s.time_completion, s.simultaneos_completion
                                       ) AS max_combinations_available
                    FROM TMP_DATA s
                    WHERE simultaneos_completion > 1
                    AND offer_receive_stack = simultaneos_completion
                ) s
                WHERE s.amount_combinations_found <= s.max_combinations_available
            ),
            NON_SIMULTANEOS AS (
                SELECT ns.customer_id, ns.offer_id, ns.time_completion, ns.time_receive, ns.time_for_completion
                FROM TMP_DATA ns
                WHERE simultaneos_completion = 1
                AND simultaneos_offers = 1
            ),
            TMP_REC_COM AS (
                SELECT * FROM NON_SIMULTANEOS
                UNION
                SELECT * FROM SIMULTANEOS
            )
            SELECT * FROM TMP_REC_COM
            '''
        self._transcript_view_query = '''
            WITH TMP_DATA AS (
                SELECT *,
                       ROW_NUMBER() OVER(
                           PARTITION BY gt.customer_id, gt.offer_id, gt.time_view
                           ORDER BY time_for_view DESC) AS multiple_views
                FROM (
                    SELECT dov2.*,
                           dor2.time_receive,
                           (dov2.time_view - dor2.time_receive) AS time_for_view
                    FROM (
                        SELECT *,
                               COUNT() OVER(
                                   PARTITION BY dov2.customer_id, dov2.offer_id, dov2.time_view
                                   ) AS simultaneos_view
                        FROM dov2
                    ) dov2
                    LEFT JOIN dor2
                    ON dov2.customer_id = dor2.customer_id
                    AND dov2.offer_id = dor2.offer_id
                    AND dov2.time_view >= dor2.time_receive
                    AND dov2.time_view <= dor2.time_limit
                ) gt
            )
            SELECT td.customer_id,
                   td.offer_id,
                   td.time_receive,
                   td.time_view,
                   Null AS time_completion,   
                   Null AS time_for_completion,
                   td.time_for_view,
                   'offer_viewed' AS general_journey,
                   Null AS amount
            FROM TMP_DATA td
            WHERE td.simultaneos_view = td.multiple_views
            AND time_receive >= 0
            '''
        self._status_function(family = "Data Import", msg = "Importation of Data")

    def _status_function(self, family:str, msg:str):
        current_formated_time = datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
        return "[{}] - [{}] {} - Done!".format(current_formated_time, family, msg)

    def _check_completion(self, view, completed, trans_type):
        if trans_type == 'informational':
            return 1 if not np.isnan(view) else 0
        else:
            return 1 if not np.isnan(view) and completed is not None else 0

    def feature_engineering_basic(self):
        """
        A feature engineering is done here in order to extra info, such as:
        - *year* of membership
        - *month* of membership
        - *day* of membership
        Besides that an ETL is performed to extract a tabular data.
        """
        self.profile['year'] = self.profile.became_member_on.apply(lambda x: x//10000)
        self.profile['month'] = self.profile.became_member_on.apply(lambda x: (x - (x//10000)*10000)//100)
        self.profile['day'] = self.profile.became_member_on.apply(lambda x: x%100)
        self.profile.drop(['became_member_on'], inplace = True, axis = 1)
        self.profile.rename(columns = {'id': 'customer_id'}, inplace = True)
        print(self._status_function(family = "Basic Feature Engineering", msg = "Profile"))

        self.transcript['amount'] = self.transcript.value.apply(lambda x: x.get('amount', None))
        self.transcript['offer_id'] = self.transcript.value.apply(lambda x: x.get('offer_id', None) if 'offer_id' in x.keys() else x.get('offer id', None))
        self.transcript['premium'] = self.transcript.value.apply(lambda x: x.get('reward', None))
        self.transcript.drop(['value'], inplace = True, axis = 1)
        self.transcript.rename(columns = {'person': 'customer_id'}, inplace = True)
        print(self._status_function(family = "Basic Feature Engineering", msg = "Transcript"))

        channels = pd.get_dummies(self.portfolio.channels.apply(pd.Series).stack()).sum(level=0)
        self.portfolio = pd.concat([self.portfolio, channels], axis=1).drop(['channels'], axis = 1)
        self.portfolio.rename(columns = {'id': 'offer_id'}, inplace = True)
        self.portfolio.duration = self.portfolio.duration.apply(lambda x: x * 24)
        print(self._status_function(family = "Basic Feature Engineering", msg = "Portfolio"))

    def reverse_engineering(self):
        """
        A reverse engeneering is done here in order to build the pipeline of the customer journey.
        1) Receive Offer -> View Offer -> Buy Offer
        2) Receive Offer -> Buy Offer
        3) Receive Offer -> View Offer
        4) Receive Offer
        5) Transaction Without Offer
        """
        gathered_table = self.transcript.merge(self.profile, on = 'customer_id', how = 'left') \
                                        .merge(self.portfolio, on = 'offer_id', how = 'left')

        self._df_offer_received = gathered_table[gathered_table['event'] == 'offer received'][['customer_id', 'time', 'offer_id', 'duration']]
        self._df_offer_completed = gathered_table[gathered_table['event'] == 'offer completed'][['customer_id', 'time', 'offer_id']]
        self._df_offer_viewed = gathered_table[gathered_table['event'] == 'offer viewed'][['customer_id', 'time', 'offer_id']]
        self._df_transactioned = gathered_table[gathered_table['event'] == 'transaction'][['customer_id', 'time', 'offer_id', 'amount']]
        self._df_offer_received['time_limit'] = self._df_offer_received['time'] + self._df_offer_received['duration']
        print(self._status_function(family = "Reverse Engineering", msg = "Transactions"))

    def completed_query(self):
        """
        Query tables through sql in order to build the paths (Receive Offer -> View Offer -> Buy Offer)
        """
        self._df_offer_received.to_sql('dor', self._conn, index=False)
        self._df_offer_completed.to_sql('doc', self._conn, index=False)

        self._query_data_transcript_completed = pd.read_sql_query(self._transcript_completed_query, self._conn)
        print(self._status_function(family = "Build Query", msg = "Offers Completed"))

    def processing_views(self):
        """
        Since it is kinda tricky to build the full pipeline through sql, *View Offer* is added with a build-in function
        (A client can have more than one possibilities between the bridge Receive and Buy, can receive the same offer more than one time)
        e.g.:
        - Receive: 451 and 500
        - View: 512 and 520 
        - Buy: 541 and 620
        (Also uploaded to S3 in order to avoid any loss of local data)
        """
        if 'tcompleted_backup.csv' in os.listdir('backup'):
            self.transcript_completed = pd.read_csv('backup/tcompleted_backup.csv')
            print(self._status_function(family = "Processing Views", msg = "Back-up Retrieved"))
        else:
            self.transcript_completed = self.finding_view_time(self._query_data_transcript_completed, self._df_offer_viewed)
            self.transcript_completed.to_csv('backup/tcompleted_backup.csv', index = False)
            self._s3.upload_file('backup/tcompleted_backup.csv', self.bucket, 'backup/tcompleted_backup.csv')
            print(self._status_function(family = "Processing Views", msg = "Offers Enrich Views Completed"))

    def finding_view_time(self, raw_dataset, views):
        """
        Build-in function to fill the View time between Receive and Buy in the Completed Table
        """
        raw_dataset_copy = raw_dataset.copy()
        views_copy = views.copy()
        all_rows = len(raw_dataset_copy)
        raw_dataset_copy.insert(loc = 4, column = 'time_view', value = None)
        for i, j in enumerate(zip(raw_dataset_copy['customer_id'].values,\
                                  raw_dataset_copy['offer_id'].values,\
                                  raw_dataset_copy['time_completion'].values,\
                                  raw_dataset_copy['time_receive'].values)):
            idx, cid, oid, tcid, trid = i, j[0], j[1], j[2], j[3]
            psbt = list(views_copy[(views_copy.customer_id == cid) &\
                                   (views_copy.offer_id == oid) &\
                                   (views_copy.time >= trid) &\
                                   (views_copy.time <= tcid)].index)
            if len(psbt) == 0:
                continue
            elif len(psbt) == 1:
                rowid = psbt[0]
                raw_dataset_copy.loc[i, 'time_view'] = views_copy.loc[rowid, 'time']
                views_copy.drop(index = rowid, inplace = True)
            else:
                rowid = np.random.choice(psbt,1)[0]
                raw_dataset_copy.loc[i, 'time_view'] = views_copy.loc[rowid, 'time']
                views_copy.drop(index = rowid, inplace = True)
            fraction = int((100*(i+1))/all_rows)
            print(f"[{fraction} %] Processing {i+1}/{all_rows}", end = '\r')
        raw_dataset_copy['time_for_view'] = raw_dataset_copy['time_completion'] - raw_dataset_copy['time_view']
        raw_dataset_copy['general_journey'] = 'offer_completed'
        return raw_dataset_copy[['customer_id', 'offer_id', 'time_receive', 'time_view', 'time_completion', 'time_for_completion', 'time_for_view', 'general_journey']]

    def retriving_monetary_tcompleted(self):
        """
        Enrich the table with monetary values to train the model
        """
        monetary_values = self.transcript[self.transcript.event == 'offer completed']\
                            [['customer_id', 'offer_id', 'time', 'premium']]\
                              .drop_duplicates()\
                             .rename(columns = {'time': 'time_completion', 'premium': 'amount'})

        self.transcript_completed = self.transcript_completed.merge(monetary_values, \
                                                          on = ['customer_id', 'offer_id', 'time_completion'], \
                                                          how = 'left')
        print(self._status_function(family = "Processing Completed", msg = "Adding Amount Value"))

    def build_received_viewed_table(self):
        """
        Build the Receive -> View journey removing the Receive -> View -> Buy journey
        """
        left_anti_view = self.transcript_completed[['customer_id', 'offer_id', 'time_view']]\
                             .dropna(subset=['time_view'])
        left_anti_view['count'] = 1

        left_anti_receive = self.transcript_completed[['customer_id', 'offer_id', 'time_receive']]\
                                .dropna(subset=['time_receive'])
        left_anti_receive['count'] = 1

        self._only_view = self._df_offer_viewed.rename(columns = {'time': 'time_view'})\
                             .merge(left_anti_view,\
                                    on = ['customer_id', 'offer_id', 'time_view'],\
                                    how = 'outer')\
                             .query('count != 1')\
                             .drop(['count'], axis = 1)

        self._only_receive = self._df_offer_received.rename(columns = {'time': 'time_receive'})\
                                .merge(left_anti_receive,\
                                       on = ['customer_id', 'offer_id', 'time_receive'],\
                                       how = 'outer')\
                                .query('count != 1')\
                                .drop(['count'], axis = 1)
        self._only_view.to_sql('dov2', self._conn, index=False)
        self._only_receive.to_sql('dor2', self._conn, index=False)
        self.transcript_viewed= pd.read_sql_query(self._transcript_view_query, self._conn)
        print(self._status_function(family = "Processing Views", msg = "Building Views Table"))

    def build_received_table(self):
        """
        Build the Receive journey removing the Receive -> View and Receive -> View -> Buy journey
        """
        left_anti_view_receive = self.transcript_viewed[['customer_id', 'offer_id', 'time_receive']].dropna(subset=['time_receive'])
        left_anti_view_receive['count'] = 1

        only_receive_view = self._only_receive\
                                .merge(left_anti_view_receive,\
                                       on = ['customer_id', 'offer_id', 'time_receive'],\
                                       how = 'outer')\
                                .query('count != 1')\
                                .drop(['count'], axis = 1)

        self.transcript_received = only_receive_view[['customer_id', 'offer_id', 'time_receive']]
        self.transcript_received[['time_view', 'time_completion', 'time_for_completion', 'time_for_view']] = None
        self.transcript_received['general_journey'] = 'offer_received'
        self.transcript_received['amount'] = None
        print(self._status_function(family = "Processing Receive", msg = "Building Receive Table"))

    def build_transactional_table(self):
        """
        Build the Transactional journey just filtering the main dataset
        """
        transcript_transactioned = self._df_transactioned.rename(columns = {'time': 'time_completion'})
        transcript_transactioned['time_receive'] = transcript_transactioned['time_completion']
        transcript_transactioned['time_view'] = transcript_transactioned['time_completion']
        transcript_transactioned[['time_for_completion', 'time_for_view']] = 0
        transcript_transactioned['general_journey'] = 'transactioned'
        self.transcript_transactioned = transcript_transactioned[self.transcript_received.columns]
        print(self._status_function(family = "Processing Transaction", msg = "Building Transactional Table"))

    def gathered_all_tables_fe(self):
        """
        Gathering all the tables with all the journeys build before in order to develop the model
        """
        self._all_offers_n_transactions_gathered = pd.concat(\
            [self.transcript_completed, self.transcript_viewed, self.transcript_received, self.transcript_transactioned], axis = 0)\
                                     .sample(frac = 1)\
                                     .reset_index()\
                                     .drop(['index'], axis = 1)
        self.complete_table = self._all_offers_n_transactions_gathered\
                                  .merge(self.profile, on = 'customer_id', how = 'left')\
                                  .merge(self.portfolio, on = 'offer_id', how = 'left')

        cols_zero = ['email', 'mobile', 'social', 'web', 'amount', 'reward', 'difficulty', 'duration']
        cols_no_offer = ['offer_type', 'offer_id']
        self.complete_table.loc[:, cols_zero] = self.complete_table.loc[:, cols_zero].fillna(0)
        self.complete_table.loc[:, cols_no_offer] = self.complete_table.loc[:, cols_no_offer].fillna('no_offer')

        print(self._status_function(family = "Processing Dataset", msg = "Building Gathered Table"))

    def feature_engineering_enhanced(self, transactional_cycle = 7 * 24):
        """
        A feature to be evaluated will be the conversion of each offer type (including transactions). Two important steps/assumptions are done here:
        1) In order to be able to measure a conversion for transaction, a lifecycle of it needs to be considered (default = 7 days = 168 hours);
        (e.g. If I have a period greater than 168 hours without doing any transactions, it will be added as a miss opportunity, like an offer
        that I have only received and not completed, so I don't have as conversion always 100 % for transactions);
        2) As a cold start for conversion, that is set as 0.5 (50 %), in order to not bias the propensity of the first conversion.
        """
        # Definition of a transactional lifecycle in order to use as a feature transaction convertional, otherwise would be always 100 %
        cols_for_cleaned_table = ['offer_type', 'gender', 'age', 'income', 'year', 'reward', 'difficulty', 'duration',\
                                  'email', 'mobile', 'social', 'web', 'cr_bogo', 'cr_transactions', 'cr_discount', 'cr_informational', 'target']
        self._tbl_enriched = None
        n_clients = len(self.complete_table.customer_id.unique())
        self.complete_table['succeed'] = self.complete_table[['time_view', 'time_completion', 'offer_type']]\
                                             .apply(lambda x: self._check_completion(x[0], x[1], x[2]), axis = 1)
        counter_keys = ['bogo', 'no_offer', 'discount', 'informational']

        for nc, i in enumerate(self.complete_table.customer_id.unique()):

            tbl = self.complete_table[self.complete_table.customer_id == i].sort_values(by = 'time_receive', ascending = True).copy()
            tbl[['r_cnt_bogo', 'r_cnt_transaction', 'r_cnt_discount', 'r_cnt_informational']] = 0
            tbl[['c_cnt_bogo', 'c_cnt_transaction', 'c_cnt_discount', 'c_cnt_informational']] = 0

            if len(tbl) > 1:
                for n, t in enumerate(tbl['time_receive'][1:]):
                    dcounter = tbl[tbl['time_receive'] < t]['offer_type'].value_counts().to_dict()
                    tbl.iloc[n+1, -8], tbl.iloc[n+1, -7], tbl.iloc[n+1, -6], tbl.iloc[n+1, -5] = \
                        [dcounter.get(k, 0) if k != 'no_offer' else max(dcounter.get(k, 0), t//transactional_cycle) for k in counter_keys]

                    dcounter = tbl[tbl['time_receive'] < t][['offer_type', 'succeed']].groupby('offer_type').sum().to_dict().get('succeed')
                    tbl.iloc[n+1, -4], tbl.iloc[n+1, -3], tbl.iloc[n+1, -2], tbl.iloc[n+1, -1] = [dcounter.get(k, 0) for k in counter_keys]
            else:
                continue
            print(f"Customers Processed: {nc+1}/{n_clients}", end = "\r")
            if self._tbl_enriched is None:
                self._tbl_enriched = tbl
            else:
                self._tbl_enriched = pd.concat([self._tbl_enriched, tbl], axis = 0)

        self._tbl_enriched['cr_bogo'] = self._tbl_enriched[['r_cnt_bogo', 'c_cnt_bogo']].apply(\
                                                            lambda x: np.clip(x[1]/x[0], a_min = 0, a_max = 1), axis = 1)

        self._tbl_enriched['cr_transactions'] = self._tbl_enriched[['r_cnt_transaction', 'c_cnt_transaction']].apply(\
                                                                            lambda x: np.clip(x[1]/x[0], a_min = 0, a_max = 1), axis = 1)

        self._tbl_enriched['cr_discount'] = self._tbl_enriched[['r_cnt_discount', 'c_cnt_discount']].apply(\
                                                                            lambda x: np.clip(x[1]/x[0], a_min = 0, a_max = 1), axis = 1)

        self._tbl_enriched['cr_informational'] = self._tbl_enriched[['r_cnt_informational', 'c_cnt_informational']].apply(\
                                                                            lambda x: np.clip(x[1]/x[0], a_min = 0, a_max = 1), axis = 1)

        self._tbl_enriched[['cr_bogo', 'cr_transactions', 'cr_discount', 'cr_informational']] =\
                self._tbl_enriched[['cr_bogo', 'cr_transactions', 'cr_discount', 'cr_informational']].fillna(0.5)

        self._tbl_enriched.rename(columns = {'succeed': 'target'}, inplace = True)
        self.tbl_enriched_cleaned = self._tbl_enriched[self._tbl_enriched['general_journey'] != 'transactioned'][cols_for_cleaned_table]

    def call_feature_engineering_enhanced(self):
        """
        Calling "feature_engineering_enhanced()" only if it wasn't already calculated since it demands a bit of time to reprocess it.
        (Also uploaded to S3 in order to avoid any loss of local data)
        """
        if 'feenhanced_backup.csv' in os.listdir('backup'):
            self.tbl_enriched_cleaned = pd.read_csv('backup/feenhanced_backup.csv')
            print(self._status_function(family = "Enhanced Feature Engineering", msg = "Back-up Retrieved"))
        else:
            self.feature_engineering_enhanced()
            self.tbl_enriched_cleaned.to_csv('backup/feenhanced_backup.csv', index = False)
            self._s3.upload_file('backup/feenhanced_backup.csv', self.bucket, 'backup/feenhanced_backup.csv')
            print(self._status_function(family = "Enhanced Feature Engineering", msg = "Adding Conversion Data"))

    def processing_eda_on_clean_data(self):
        return ProfileReport(self.tbl_enriched_cleaned)

    def fit_etl(self):
        """
        Save time and run all at once
        """
        self.feature_engineering_basic()
        self.reverse_engineering()
        self.completed_query()
        self.processing_views()
        self.retriving_monetary_tcompleted()
        self.build_received_viewed_table()
        self.build_received_table()
        self.build_transactional_table()
        self.gathered_all_tables_fe()
        self.call_feature_engineering_enhanced()

    def fit_preprocess(self):
        """
        The pipeline of preprocessing step is done in this stage. The pipe is compose by 4 flows:
        1) Handling the categorical data with one-hot-encoding;
        2) Treating continuos data with standard scaler;
        3) Handle discrete data with minmax scaler;
        4) Data that needs no treatment.
        By the end a preprocessor is saved for future use.
        """
        self._one_hot_scaler_cols = ['gender', 'year', 'offer_type']
        self._standard_scaler_cols = ['income', 'age', 'cr_bogo', 'cr_transactions', 'cr_discount', 'cr_informational']
        self._min_max_scaler_cols = ['reward', 'difficulty', 'duration']
        self._passthrough_cols = ['email', 'mobile', 'social', 'web', 'target']

        one_hot_scaler = Pipeline(\
            steps = [('oh_cols', ColumnSelector(cols=self._one_hot_scaler_cols)),\
                     ("oh_imputer", CategoricalImputer(method="most_frequent", fill_value=None)),\
                     ('oh_method', OneHotEncoder(handle_unknown="ignore", sparse=False))])

        standard_scaler = Pipeline(\
            steps=[("std_scaler_cols", ColumnSelector(cols=self._standard_scaler_cols)),\
                   ("std_scaler_impute", CategoricalImputer(method="most_frequent", fill_value = None)),\
                   ("std_scaler_method", StandardScaler())])

        min_max_scaler = Pipeline(\
            steps=[("min_max_cols", ColumnSelector(cols=self._min_max_scaler_cols)),\
                   ("min_max_imputer", CategoricalImputer(method="most_frequent", fill_value=None)),\
                   ("min_max_method", MinMaxScaler())])

        self.preprocessor = FeatureUnion(transformer_list=[\
                    ("oh_process", one_hot_scaler),\
                    ("std_process", standard_scaler),\
                    ("min_max_process", min_max_scaler)])


        X, y = self.tbl_enriched_cleaned.drop(['target'], axis = 1), self.tbl_enriched_cleaned.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.25, stratify = y)

        self.preprocessor.fit(self.X_train)
        self.tbl_enriched_cleaned_preprocess = self.preprocessor.transform(self.tbl_enriched_cleaned)
        pickle.dump(self.preprocessor, open('backup/preprocessor.pkl', 'wb'))
        self._s3.upload_file('backup/preprocessor.pkl', self.bucket, 'backup/preprocessor.pkl')
        with tarfile.open('backup/preprocessor.tar.gz', 'w:gz') as tar:
            tar.add('backup/preprocessor.pkl', arcname=os.path.basename('backup/preprocessor.pkl'))
        self._s3.upload_file('backup/preprocessor.tar.gz', self.bucket, 'backup/preprocessor.tar.gz')
        print(self._status_function(family = "Preprocessor", msg = "Generating Preprocessor"))

    def model_architecture_tuned(self):
        """
        The model is tuned here through GridSearch with cross validation in order to find the best hyperparameters to be used.
        """
        self.X_train_preprocessed = self.preprocessor.transform(self.X_train)
        self.X_test_preprocessed = self.preprocessor.transform(self.X_test)
        LGBM = LGBMClassifier(verbose=-1)
        param_grid = {
            'learning_rate': [0.01, 0.1, 0.25],
            'n_estimators': [100, 200, 300],
            'num_leaves': [20, 40, 80],
        }
        self.lgbm_tuned = GridSearchCV(
            estimator = LGBM,
            param_grid = param_grid,
            scoring = 'accuracy',
            cv = 3,
            verbose = 0
        )
        self.lgbm_tuned.fit(self.X_train_preprocessed, self.y_train)
        pickle.dump(self.lgbm_tuned, open('backup/model.pkl', 'wb'))
        self._s3.upload_file('backup/model.pkl', self.bucket, 'backup/model.pkl')
        with tarfile.open('backup/model.tar.gz', 'w:gz') as tar:
            tar.add('backup/model.pkl', arcname=os.path.basename('backup/model.pkl'))
        self._s3.upload_file('backup/model.tar.gz', self.bucket, 'backup/model.tar.gz')
        self.best_params = self.lgbm_tuned.best_params_
        self.best_model = self.lgbm_tuned.best_estimator_
        print(self._status_function(family = "Tuning", msg = "Gridsearch on the Model"))

    def call_model_architecture_tuned(self):
        """
        Training of the preprocessor and the model is called here with they are not find in the backup folder.
        """
        if 'model.pkl' in os.listdir('backup') and 'preprocessor.pkl' in os.listdir('backup'):
            with open('backup/preprocessor.pkl', 'rb') as preprocessor:
                self.preprocessor = pickle.load(preprocessor)
            with open('backup/model.pkl', 'rb') as model:
                self.lgbm_tuned = pickle.load(model)
            print(self._status_function(family = "Tuning", msg = "Model and Preprocessor Back-up Retrieved"))
        else:
            self.fit_preprocess()
            self.model_architecture_tuned()

    def full_pipeline(self):
        """
        It connects the part of the processor with the model itself in order to do only one deployment for generating the endpoint.
        """
        if 'process_model_pipeline.pkl' in os.listdir('backup'):
            with open('backup/process_model_pipeline.pkl', 'rb') as process_model_pipeline:
                self.process_model_pipeline = pickle.load(process_model_pipeline)
            print(self._status_function(family = "Pipeline", msg = "Pipeline Back-up Retrieved (Processor + Model)"))
        else:
            self.process_model_pipeline = Pipeline(steps = [('preprocessor', self.preprocessor),\
                                                            ('model_tuned', self.lgbm_tuned)])
            pickle.dump(self.process_model_pipeline, open('backup/process_model_pipeline.pkl', 'wb'))
            self._s3.upload_file('backup/process_model_pipeline.pkl', self.bucket, 'backup/process_model_pipeline.pkl')
            with tarfile.open('backup/process_model_pipeline.tar.gz', 'w:gz') as tar:
                tar.add('backup/process_model_pipeline.pkl', arcname=os.path.basename('backup/process_model_pipeline.pkl'))
            self._s3.upload_file('backup/process_model_pipeline.tar.gz', self.bucket, 'backup/process_model_pipeline.tar.gz')
            print(self._status_function(family = "Pipeline", msg = "Pipeline Completed (Processor + Model)"))

    def custom_predict(self, X):
        """
        Custom predict function with the threshold set in threshold_iteration_tuning or manually set
        [Default Setting] self.thresh_params = {'bogo': 0.50, 'discount': 0.50, 'informational': 0.50}
        It can be manually changed directly through the attribute thresh_params from the class
        """
        proba_n_types = pd.concat([X.offer_type.reset_index().drop(['index'], axis = 1),\
                                   pd.DataFrame(self.process_model_pipeline.predict_proba(X)[:, 1],\
                                                columns = ['proba'])\
                                  ], axis=1)
        custom_output = proba_n_types.apply(
                         lambda x: 1 if self.thresh_params.get(x[0]) <= x[1] else 0,
                         axis=1
                                           ).values
        return custom_output

    def save_frame_img(self, df, name, folder = 'images', width = 14, height = 6):

        ax = plt.subplot(111, frame_on=False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        fig = plt.gcf()
        fig.set_size_inches(width, height) 
        table(ax, df, loc="center")
        plt.savefig(f'{folder}/{name}.png')

    def threshold_iteration_tuning(self, step = 0.01, mode_type = 'accuracy'):
        """
        Another feature of this class is that it is feasible to tune the threshold for each offer_type seeking maximizing
        the accuracy.
        Metrics Available: 'accuracy' (default), 'f1_score', 'jaccard_score'.
        """
        assert mode_type in ['accuracy', 'f1_score', 'jaccard_score'], "Select a valid metric for tuning."
        self.thresh_evaluation = {}
        for i in self.X_train.offer_type.unique():
            best_usecase = [i, 0, 0] 
            X_filtered = self.X_train[self.X_train.offer_type == i]
            y_train = self.y_train[self.y_train.index.isin(X_filtered.index)]
            for j in np.arange(0, 1+step, step):
                self.thresh_params[i] = j
                y_pred_custom = self.custom_predict(X_filtered)

                if mode_type == 'accuracy':
                    metric_custom = round(accuracy_score(y_train, y_pred_custom), 4)
                elif mode_type == 'f1_score':
                    metric_custom = round(f1_score(y_train, y_pred_custom), 4)
                else:
                    metric_custom = round(jaccard_score(y_train, y_pred_custom), 4)

                if best_usecase[2] <= metric_custom:
                    best_usecase[1], best_usecase[2] = j, metric_custom
            self.thresh_evaluation[i] = {'step': best_usecase[1], 'value': best_usecase[2]}

        self.thresh_params['bogo'] = self.thresh_evaluation.get('bogo').get('step')
        self.thresh_params['discount'] = self.thresh_evaluation.get('discount').get('step')
        self.thresh_params['informational'] = self.thresh_evaluation.get('informational').get('step')
        print(self._status_function(family = "Tuning Threshold", msg = "Evaluating Best Threshold on Metric Selected"))

        return self.thresh_evaluation

    def measuring_basic_metrics(self, y_test, y_pred):
        cmatrix = confusion_matrix(y_test, y_pred)
        metrics = {
            'accuracy': round(accuracy_score(y_test, y_pred), 4),
            'precision': round(precision_score(y_test, y_pred), 4),
            'recall': round(recall_score(y_test, y_pred), 4),
            'f1_score': round(f1_score(y_test, y_pred), 4),
            'jaccard_score': round(jaccard_score(y_test, y_pred), 4),
            'specificity': round((cmatrix[0, 0] / (cmatrix[0, 0] + cmatrix[0, 1])), 4),
            'false_positive_rate': round((cmatrix[0, 1] / (cmatrix[0, 1] + cmatrix[0, 0])), 4),
            'false_negative_rate': round((1 - recall_score(y_test, y_pred)), 4)
        }
        tmp_metrics = pd.DataFrame(index=metrics.keys(), data=metrics.values(), columns=['value'])\
                        .reset_index()\
                        .rename(columns={'index': 'metric'})
        return tmp_metrics

    def model_evaluation_metrics(self, custom:bool):
        """
        Most of the model evaluation is done through here. Since it is a classification problem, the metrics that
        were measured are regarding it and a table is generated from it.
        Futhermore, 3 main images are generated for each offer_type and general case.
        - Confusion Matrix;
        - ROC-AUC Curve;
        - Precision-Recall Curve.
        It is possible to select custom = True, with that selected, the metrics generates are measure considering
        the custom threshold that is created through threshold_iteration_tuning.s
        """
        tmp_metrics_gathered = None

        def fig_confusion_matrix(y_true, y_pred, section):
            cm = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap = 'Greens');
            plt.title(f'confusion matrix - {section}');
            cm.figure_.savefig(f'images/confusion_matrix_{i}.png')

        def fig_roc_auc_curve(y_true, y_pred, section):
            roc_auc = roc_auc_score(y_true, y_pred)
            fpr, tpr, _ = roc_curve(y_true, y_pred)

            fig = plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC curve - {section}')
            plt.legend(loc='lower right')
            plt.show()
            fig.savefig(f'images/roc_auc_curve_{section}.png')

        def fig_precision_recall_curve(y_true, y_pred, section):
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            fig = plt.figure()
            plt.step(recall, precision, color='b', alpha=0.2, where='post')
            plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'precision-recall curve - {section}')
            plt.show()
            fig.savefig(f'images/precision_recall_curve_{section}.png')

        for i in self.X_test.offer_type.unique():
            tmp_metrics = None
            X_filtered = self.X_test[self.X_test.offer_type == i]
            y_test = self.y_test[self.y_test.index.isin(X_filtered.index)]
            y_pred = self.custom_predict(X_filtered) if custom else self.process_model_pipeline.predict(X_filtered)

            tmp_metrics = self.measuring_basic_metrics(y_test, y_pred)
            tmp_metrics['offer_type'] = i
            fig_confusion_matrix(y_test, y_pred, i)

            X_filtered = self.X_train[self.X_train.offer_type == i]
            y_train = self.y_train[self.y_train.index.isin(X_filtered.index)]
            y_pred = self.custom_predict(X_filtered) if custom else self.process_model_pipeline.predict(X_filtered)
            fig_roc_auc_curve(y_train, y_pred, i)
            fig_precision_recall_curve(y_train, y_pred, i)

            if tmp_metrics is None:
                tmp_metrics_gathered = tmp_metrics
            else:
                tmp_metrics_gathered = pd.concat([tmp_metrics_gathered, tmp_metrics], axis = 0)

        y_test = self.y_test
        y_pred = self.custom_predict(self.X_test) if custom else self.process_model_pipeline.predict(self.X_test)
        tmp_metrics = self.measuring_basic_metrics(y_test, y_pred)
        tmp_metrics['offer_type'] = 'all_offers'
        fig_confusion_matrix(y_test, y_pred, 'all_offers')
        tmp_metrics_gathered = pd.concat([tmp_metrics_gathered, tmp_metrics], axis = 0)

        y_train = self.y_train
        y_pred = self.custom_predict(self.X_train) if custom else self.process_model_pipeline.predict(self.X_train)
        fig_roc_auc_curve(y_train, y_pred, 'all_offers')
        fig_precision_recall_curve(y_train, y_pred, 'all_offers')

        self.basic_metrics_table = pd.pivot_table(tmp_metrics_gathered, values="value", index=["metric"], columns=["offer_type"])
        self.save_frame_img(self.basic_metrics_table, 'model_metrics')
        print(self._status_function(family = "Evaluation", msg = "Performing Evaluation of the Model"))

        return

    def plot_shap(self, n_samples = 100):
        """
        Plot of the SHAP summary in order to check the importance of features, in order to interpret the problem.
        Besides that, it is possible to perform feature selection through it. Mainly to see if it is feasible to
        drop any variable non significant, simplifying the problem itself.
        """
        if n_samples > 100:
            print("[WARNING] Values greater than 100 can spend a considerable amount of memory and time.")

        def model_predict(data_asarray):
            data_asframe = pd.DataFrame(data_asarray, columns=list(self.X_train.columns))
            return self.process_model_pipeline.predict_proba(data_asframe)

        shap.initjs()
        train_sample, test_sample = self.X_train.sample(n_samples), self.X_test.sample(n_samples)
        shap_kernel_explainer = shap.KernelExplainer(model_predict, train_sample)
        shap_values_single = shap_kernel_explainer.shap_values(test_sample)
        shap.summary_plot(np.array(shap_values_single[0]), test_sample, plot_type = 'bar')

    def business_rules(self):
        """
        Business rules are here applied, the 4 most variables were selected by the feature importance evaluation in
        order to make simple filters, every variable has a condition that raises a flag, like, income above 70k, or
        offered through social, a specific year that has started the membership, or a x amount of reward.
        - People that have 2 flags or more raised will be set as buyer;
        - People with 1 or no flags raised will bet set as non-buyer.
        """
        self.business_table = {}
        categorical_variables = ['reward', 'year', 'social']
        continuos_variables = ['income']
        for i in categorical_variables:
            self.business_table[i] = {}
            for offer in self.tbl_enriched_cleaned.offer_type.unique():
                aggregating_table = self.tbl_enriched_cleaned[self.tbl_enriched_cleaned.offer_type == offer]
                aggregating_table = aggregating_table[[i, 'target']]\
                                    .groupby(i)\
                                    .mean()\
                                    .apply(lambda x: 1 if x[-1] >= 0.50 else 0.00, axis = 1)\
                                    .reset_index()\
                                    .rename(columns={0: 'flag'})
                self.business_table[i][offer] = {int(k): int(v) for (k,v) in zip(aggregating_table[i].values, aggregating_table['flag'].values)}

        for i in continuos_variables:
            aggregating_table = self.tbl_enriched_cleaned[[i, 'target']]
            aggregating_table['quantile'] = pd.qcut(aggregating_table[i], q=3, labels=False) + 1
            aggregating_table_grouped = aggregating_table.groupby(['quantile']).agg({'income': [min, max], 'target': np.mean}).reset_index()
            aggregating_table_grouped.columns = ['quantile', 'min', 'max', 'flag']
            aggregating_table_grouped['flag'] = aggregating_table_grouped['flag'].apply(lambda x: 1 if x >= 0.50 else 0.00)
            aggregating_table_grouped.drop(['quantile'], axis = 1, inplace = True)
            min_value = aggregating_table_grouped[aggregating_table_grouped['flag'] == 1]['min'].min()
            self.business_table[i] = min_value
        print(self._status_function(family = "Evaluation", msg = "Building Business Rules Dictionary"))

    def application_business_rules(self):

        def read_rules(r, y, s, i, o):

            flag_thresh = 1
            reward_falg = self.business_table.get('reward').get(o).get(r)
            year_flag = self.business_table.get('year').get(o).get(y)
            social_flag = self.business_table.get('social').get(o).get(s)
            income_limit = 0 if i < self.business_table.get('income') else 1

            return 1 if (reward_falg + year_flag + social_flag + income_limit) >= flag_thresh else 0

        self.tbl_enriched_cleaned_business_rules_applied = self.tbl_enriched_cleaned.copy()
        self.tbl_enriched_cleaned_business_rules_applied['flags'] =\
            self.tbl_enriched_cleaned_business_rules_applied[list(self.business_table.keys()) + ['offer_type']]\
                .apply(lambda x: read_rules(x[0], x[1], x[2], x[3], x[4]), axis = 1)

        self.business_rules_metrics = None
        for i in self.tbl_enriched_cleaned_business_rules_applied.offer_type.unique():
            y_test = self.tbl_enriched_cleaned_business_rules_applied.query(f"offer_type == '{i}'")['target']
            y_pred = self.tbl_enriched_cleaned_business_rules_applied.query(f"offer_type == '{i}'")['flags']
            tmp_tbl = self.measuring_basic_metrics(y_test, y_pred)
            tmp_tbl['offer_type'] = i
            if self.business_rules_metrics is None:
                self.business_rules_metrics = tmp_tbl
            else:
                self.business_rules_metrics = pd.concat([self.business_rules_metrics, tmp_tbl], axis = 0)
        y_test = self.tbl_enriched_cleaned_business_rules_applied['target']
        y_pred = self.tbl_enriched_cleaned_business_rules_applied['flags']
        tmp_tbl = self.measuring_basic_metrics(y_test, y_pred)
        tmp_tbl['offer_type'] = 'all_offers'
        self.business_rules_metrics = pd.concat([self.business_rules_metrics, tmp_tbl], axis = 0)
        self.business_rules_metrics = pd.pivot_table(self.business_rules_metrics, values="value", index=["metric"], columns=["offer_type"])
        self.save_frame_img(self.business_rules_metrics, 'business_metrics')

        print(self._status_function(family = "Evaluation", msg = "Evaluating Business Rules"))

def model_fn(model_dir):
    """
    Function that is necessary for calling the model by the time of deployment and creating the endpoint on the notebook.
    (Note that the function is not inside the class)
    """
    with open(os.path.join(model_dir, "process_model_pipeline.pkl"), "rb") as f:
        model = pickle.load(f)
    return model


def input_fn(input_data, content_type):
    if content_type == 'text/csv':
        df = pd.read_csv(StringIO(input_data))
        return df
    elif content_type == 'application/json':
        df = pd.read_json(input_data)
        return df
    else:
        raise ValueError(f"{content_type} currently not supported for inference!")


def predict_fn(input_data, model):
    # probabilities = model.predict_proba(input_data)[:, 1]
    probabilities = model.predict(input_data)
    return probabilities


def output_fn(predictions, content_type):
    assert content_type == 'application/json', "Only content type 'application/json' is supported!"
    response = {
        "response": predictions.tolist()
    }
    return json.dumps(response)


def save_model(model, model_dir):
    """
    Saves the model in the container environment folder by the time of deployment.
    (Note that the function is not inside the class)
    """
    path = os.path.join(model_dir, "process_model_pipeline.pkl")
    pickle.dump(model, open(path, 'wb'))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    SP = StarbucksProject()
    SP.fit_etl()
    SP.fit_preprocess()
    SP.call_model_architecture_tuned()
    SP.full_pipeline()
    SP.threshold_iteration_tuning()
    SP.model_evaluation_metrics()
    SP.business_rules()
    SP.application_business_rules()

    save_model(SP.process_model_pipeline, parser.parse_args().model_dir)