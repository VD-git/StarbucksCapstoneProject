import pandas as pd
import numpy as np
import math
import json
import sqlite3
import datetime
import os
import logging

class StarbucksProject():

    def __init__(self):
        """
        While the initiation of the class, the data and connections are already done, name of the files and folders are not supposed to be changed.
        """
        self.portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
        self.profile = pd.read_json('data/profile.json', orient='records', lines=True)
        self.transcript = pd.read_json('data/transcript.json', orient='records', lines=True)
        self._conn = sqlite3.connect(':memory:')
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

    def feature_engineering(self):
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
        print(self._status_function(family = "Feature Engineering", msg = "Profile"))

        self.transcript['amount'] = self.transcript.value.apply(lambda x: x.get('amount', None))
        self.transcript['offer_id'] = self.transcript.value.apply(lambda x: x.get('offer_id', None) if 'offer_id' in x.keys() else x.get('offer id', None))
        self.transcript['premium'] = self.transcript.value.apply(lambda x: x.get('reward', None))
        self.transcript.drop(['value'], inplace = True, axis = 1)
        self.transcript.rename(columns = {'person': 'customer_id'}, inplace = True)
        print(self._status_function(family = "Feature Engineering", msg = "Transcript"))

        channels = pd.get_dummies(self.portfolio.channels.apply(pd.Series).stack()).sum(level=0)
        self.portfolio = pd.concat([self.portfolio, channels], axis=1).drop(['channels'], axis = 1)
        self.portfolio.rename(columns = {'id': 'offer_id'}, inplace = True)
        self.portfolio.duration = self.portfolio.duration.apply(lambda x: x * 24)
        print(self._status_function(family = "Feature Engineering", msg = "Portfolio"))

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
        """
        if 'tcompleted_backup.csv' in os.listdir():
            self.transcript_completed = pd.read_csv('tcompleted_backup.csv')
            print(self._status_function(family = "Processing Views", msg = "Back-up Retrieved"))
        else:
            self.transcript_completed = self.finding_view_time(self._query_data_transcript_completed, self._df_offer_viewed)
            self.transcript_completed.to_csv('tcompleted_backup.csv', index = False)
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
        transcript_transactioned[['time_receive', 'time_view', 'time_for_completion', 'time_for_view']] = None
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

        self.complete_table.to_csv('preprocessed_data.csv', index = False, sep = ';')
        print(self._status_function(family = "Processing Datase", msg = "Building Gathered Table"))

    def fit_preprocess(self):
        """
        Save time and run all at once
        """
        self.feature_engineering()
        self.reverse_engineering()
        self.completed_query()
        self.processing_views()
        self.retriving_monetary_tcompleted()
        self.build_received_viewed_table()
        self.build_received_table()
        self.build_transactional_table()
        self.gathered_all_tables_fe()


if __name__ == '__main__':

    SP = StarbucksProject()
    SP.fit_preprocess()