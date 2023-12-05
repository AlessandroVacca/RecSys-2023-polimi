import numpy as np
import scipy.sparse as sps
import pandas as pd
import tqdm
from xgboost import XGBRanker
from Recommenders.BaseRecommender import BaseRecommender

class XgBoostEnsembler(BaseRecommender):
    """
    This class is used to create an ensemble model using XGBoost.
    It inherits from the BaseRecommender class.
    """
    RECOMMENDER_NAME = "XgBoostEnsembler"

    def __init__(self, URM_train, URM_val, recommenders: dict, verbose=True):
        """
        Constructor for the XgBoostEnsembler class.

        Parameters:
        URM_train (scipy.sparse matrix): The user-item matrix for training.
        URM_val (scipy.sparse matrix): The user-item matrix for validation.
        recommenders (dict): A dictionary of recommenders.
        verbose (bool): A flag used to print detailed logs. Default is True.
        """
        super().__init__(URM_train, verbose=verbose)
        self.URM_val = URM_val
        self.recommenders = recommenders
        self.XGB_model = self._init_XGB_model()

    def _init_XGB_model(self):
        """
        Initializes the XGBoost model with predefined parameters.

        Returns:
        XGBRanker: The initialized XGBoost model.
        """
        return XGBRanker(
            objective='rank:pairwise',
            n_estimators=1000,
            learning_rate=0.19823429576094637,
            reg_alpha=47,
            reg_lambda=0.313,
            max_depth=3,
            verbosity=0,
            booster="gbtree",
            colsample_bytree=0.6113704247857885,
            gamma=8.964184693722684,
            min_child_weight=7.0,
            tree_method="hist"
        )

    def _create_training_dataframe(self, n_users, cutoff):
        """
        Creates a training dataframe with user IDs and their corresponding item recommendations.

        Parameters:
        n_users (int): The number of users.
        cutoff (int): The number of items to recommend.

        Returns:
        DataFrame: The training dataframe.
        """
        training_dataframe = pd.DataFrame(index=range(0, n_users), columns=["ItemID"])
        training_dataframe.index.name = 'UserID'
        for user_id in tqdm.tqdm(range(n_users)):
            recommendations = self.recommenders["SLIM_ELASTIC"].recommend(user_id, cutoff=cutoff)
            training_dataframe.loc[user_id, "ItemID"] = recommendations
        return training_dataframe.explode("ItemID")

    def _populate_training_dataframe(self, training_dataframe, n_users):
        """
        Populates the training dataframe with item scores from each recommender.

        Parameters:
        training_dataframe (DataFrame): The training dataframe.
        n_users (int): The number of users.

        Returns:
        DataFrame: The populated training dataframe.
        """
        for user_id in tqdm.tqdm(range(n_users)):
            for rec_label, rec_instance in self.recommenders.items():
                item_list = training_dataframe.loc[user_id, "ItemID"].values.tolist()
                all_item_scores = rec_instance._compute_item_score([user_id], items_to_compute=item_list)
                training_dataframe.loc[user_id, rec_label] = all_item_scores[0, item_list]
        return training_dataframe

    def fit(self, cutoff=35):
        """
        Trains the XGBoost model.

        Parameters:
        cutoff (int): The number of items to recommend. Default is 35.
        """
        n_users, n_items = self.URM_train.shape
        training_dataframe = self._create_training_dataframe(n_users, cutoff)
        training_dataframe = self._populate_training_dataframe(training_dataframe, n_users)
        X_train = training_dataframe.drop(columns=["Label", "UserID", "ItemID"])
        y_train = training_dataframe["Label"]
        groups = training_dataframe.groupby("UserID").size().values
        self.XGB_model.fit(X_train, y_train, group=groups, verbose=True)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        Computes the item scores for a given array of user IDs.

        Parameters:
        user_id_array (array): The array of user IDs.
        items_to_compute (array): The array of items to compute scores for. Default is None.

        Returns:
        array: The array of item scores.
        """
        n_users, n_items = self.URM_train.shape
        training_dataframe = self._create_training_dataframe(user_id_array, 35)
        training_dataframe = self._populate_training_dataframe(training_dataframe, user_id_array)
        X_train = training_dataframe.drop(columns=["Label", "UserID", "ItemID"])
        predictions = self.XGB_model.predict(X_train)
        return self._rerank_items(predictions, training_dataframe, user_id_array, n_items)

    def _rerank_items(self, predictions, training_dataframe, user_id_array, n_items):
        """
        Reranks the items based on the predictions from the XGBoost model.

        Parameters:
        predictions (array): The array of predictions from the XGBoost model.
        training_dataframe (DataFrame): The training dataframe.
        user_id_array (array): The array of user IDs.
        n_items (int): The number of items.

        Returns:
        array: The array of reranked item scores.
        """
        reranked_dataframe = training_dataframe.copy()
        reranked_dataframe['rating_xgb'] = pd.Series(predictions, index=reranked_dataframe.index)
        reranked_dataframe = reranked_dataframe.sort_values(['UserID', 'rating_xgb'], ascending=[True, False])
        result = []
        for id in tqdm.tqdm(user_id_array):
            items_scores = np.ones(n_items)*(-100000000)
            items_scores[reranked_dataframe.loc[reranked_dataframe['UserID'] == id].ItemID.values.astype(int)] = reranked_dataframe.loc[reranked_dataframe['UserID'] == id].rating_xgb.values
            result.append(items_scores)
        return np.array(result)