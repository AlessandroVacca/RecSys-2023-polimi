import numpy as np
import scipy.sparse as sps
import pandas as pd
import tqdm
from joblib import Parallel, delayed
from xgboost import XGBRanker

from Recommenders.BaseRecommender import BaseRecommender


class XgBoostEnsembler(BaseRecommender):
    RECOMMENDER_NAME = "LinearZScoreNormalizedHybridRecommender"

    def __init__(self, URM_train, URM_val, recommenders: dict, verbose=True):
        self.URM_val = URM_val
        self.RECOMMENDER_NAME = 'XgBoostEnsembler'

        super(XgBoostEnsembler, self).__init__(URM_train, verbose=verbose)

        self.recommenders = recommenders

        # XGBOOST
        n_estimators = 1000
        learning_rate = 0.19823429576094637
        reg_alpha = 47
        reg_lambda = 0.313
        max_depth = 3
        max_leaves = 0
        grow_policy = "depthwise"
        objective = "pairwise"
        booster = "gbtree"
        use_user_profile = False
        random_seed = None

        colsample_bytree = 0.6113704247857885
        gamma = 8.964184693722684
        min_child_weight = 7.0

        self.XGB_model = XGBRanker(objective='rank:{}'.format(objective),
                              n_estimators=int(n_estimators),
                              random_state=random_seed,
                              learning_rate=learning_rate,
                              reg_alpha=reg_alpha,
                              reg_lambda=reg_lambda,
                              max_depth=int(max_depth),
                              max_leaves=int(max_leaves),
                              grow_policy=grow_policy,
                              verbosity=0,  # 2 if self.verbose else 0,
                              booster=booster,
                              colsample_bytree=colsample_bytree,
                              gamma=gamma,
                              min_child_weight=min_child_weight,
                              tree_method="hist"
                              )

    def fit(self, cutoff=35):

        n_users, n_items = self.URM_train.shape
        training_dataframe = pd.DataFrame(index=range(0, n_users), columns=["ItemID"])
        training_dataframe.index.name = 'UserID'
        training_dataframe = training_dataframe.explode("ItemID")
        print("CREATING THE DATAFRAME")
        for user_id in tqdm.tqdm(range(n_users)):
            recommendations = self.recommenders["SLIM_ELASTIC"].recommend(user_id, cutoff=cutoff)
            training_dataframe.loc[user_id, "ItemID"] = recommendations

        training_dataframe = training_dataframe.explode("ItemID")

        urm_validation_coo = sps.coo_matrix(self.URM_val)

        correct_recommendations = pd.DataFrame({"UserID": urm_validation_coo.row,
                                                "ItemID": urm_validation_coo.col})
        training_dataframe = pd.merge(training_dataframe, correct_recommendations, on=['UserID', 'ItemID'], how='left',
                                      indicator='Exist')

        training_dataframe["Label"] = training_dataframe["Exist"] == "both"
        training_dataframe.drop(columns=['Exist'], inplace=True)
        print("POPULATING THE DATAFRAME")

        training_dataframe = training_dataframe.set_index('UserID')

        for user_id in tqdm.tqdm(range(n_users)):
            for rec_label, rec_instance in self.recommenders.items():

                item_list = training_dataframe.loc[user_id, "ItemID"].values.tolist()

                all_item_scores = rec_instance._compute_item_score([user_id], items_to_compute = item_list)

                training_dataframe.loc[user_id, rec_label] = all_item_scores[0, item_list]

        training_dataframe = training_dataframe.reset_index()
        training_dataframe = training_dataframe.rename(columns = {"index": "UserID"})

        item_popularity = np.ediff1d(sps.csc_matrix(self.URM_train).indptr)
        training_dataframe['item_popularity'] = item_popularity[training_dataframe["ItemID"].values.astype(int)]

        user_popularity = np.ediff1d(sps.csr_matrix(self.URM_train).indptr)
        training_dataframe['user_profile_len'] = user_popularity[training_dataframe["UserID"].values.astype(int)]
        training_dataframe = training_dataframe.sort_values("UserID").reset_index()
        training_dataframe.drop(columns=['index'], inplace=True)
        groups = training_dataframe.groupby("UserID").size().values

        print("START TRAINING XGBOOSS")


        training_dataframe.ItemID = training_dataframe.ItemID.astype("int64")

        X_train = training_dataframe.drop(columns=["Label", "UserID", "ItemID"])
        y_train = training_dataframe["Label"]

        self.XGB_model.fit(X_train,
                      y_train,
                      group=groups,
                      verbose=True)
    def _compute_item_score(self, user_id_array, items_to_compute=None):
        cutoff = 35
        n_users, n_items = self.URM_train.shape

        training_dataframe = pd.DataFrame(index=user_id_array, columns=["ItemID"])
        training_dataframe.index.name = 'UserID'
        training_dataframe = training_dataframe.explode("ItemID")
        print("CREATING THE DATAFRAME")
        for user_id in tqdm.tqdm(user_id_array):
            recommendations = self.recommenders["SLIM_ELASTIC"].recommend(user_id, cutoff=cutoff)
            training_dataframe.loc[user_id, "ItemID"] = recommendations

        training_dataframe = training_dataframe.explode("ItemID")

        urm_validation_coo = sps.coo_matrix(self.URM_val)

        correct_recommendations = pd.DataFrame({"UserID": urm_validation_coo.row,
                                                "ItemID": urm_validation_coo.col})
        training_dataframe = pd.merge(training_dataframe, correct_recommendations, on=['UserID', 'ItemID'], how='left',
                                      indicator='Exist')

        training_dataframe["Label"] = training_dataframe["Exist"] == "both"
        training_dataframe.drop(columns=['Exist'], inplace=True)
        print("POPULATING THE DATAFRAME")

        training_dataframe = training_dataframe.set_index('UserID')

        for user_id in tqdm.tqdm(user_id_array):
            for rec_label, rec_instance in self.recommenders.items():
                item_list = training_dataframe.loc[user_id, "ItemID"].values.tolist()

                all_item_scores = rec_instance._compute_item_score([user_id], items_to_compute=item_list)

                training_dataframe.loc[user_id, rec_label] = all_item_scores[0, item_list]

        training_dataframe = training_dataframe.reset_index()
        training_dataframe = training_dataframe.rename(columns={"index": "UserID"})

        item_popularity = np.ediff1d(sps.csc_matrix(self.URM_train).indptr)
        training_dataframe['item_popularity'] = item_popularity[training_dataframe["ItemID"].values.astype(int)]

        user_popularity = np.ediff1d(sps.csr_matrix(self.URM_train).indptr)
        training_dataframe['user_profile_len'] = user_popularity[training_dataframe["UserID"].values.astype(int)]
        training_dataframe = training_dataframe.sort_values("UserID").reset_index()
        training_dataframe.drop(columns=['index'], inplace=True)
        groups = training_dataframe.groupby("UserID").size().values

        X_train = training_dataframe.drop(columns=["Label", "UserID", "ItemID"])

        predictions = self.XGB_model.predict(X_train)

        reranked_dataframe = training_dataframe.copy()
        reranked_dataframe['rating_xgb'] = pd.Series(predictions, index=reranked_dataframe.index)

        reranked_dataframe = reranked_dataframe.sort_values(['UserID', 'rating_xgb'], ascending=[True, False])
        result = []
        for id in tqdm.tqdm(user_id_array):
            items_scores = np.ones(n_items)*(-100000000)
            items_scores[reranked_dataframe.loc[reranked_dataframe['UserID'] == id].ItemID.values.astype(int)] = reranked_dataframe.loc[reranked_dataframe['UserID'] == id].rating_xgb.values
            result.append(items_scores)

        result = np.array(result)
        return result

