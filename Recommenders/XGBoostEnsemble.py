import numpy as np
import scipy.sparse as sps
import pandas as pd
import sklearn.svm
import tqdm
import xgboost
from sklearn.model_selection import train_test_split
from xgboost import XGBRanker
from Recommenders.BaseRecommender import BaseRecommender
from xgboost import plot_importance


class XgBoostEnsembler(BaseRecommender):
    """
    This class is used to create an ensemble model using XGBoost.
    It inherits from the BaseRecommender class.
    """
    RECOMMENDER_NAME = "XgBoostEnsembler"

    def __init__(self, URM_train, URM_val, recommenders: dict, recommenders_generate_data: dict,
                 internal_cutoff_xgboost=10, verbose=True):
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
        self.internal_cutoff_xgboost = internal_cutoff_xgboost
        self.recommenders_generate_data = recommenders_generate_data
        self.training_dataframe = None

    def _init_XGB_model(self, **xgb_ranker_params)
        """
        Initializes the XGBoost model with predefined parameters.

        Returns:
        XGBRanker: The initialized XGBoost model.
        """
        return XGBRanker(
            **xgb_ranker_params,

            # eta=0.0506,
            # booster="gbtree",
            # colsample_bytree=0.6113704247857885,
            # gamma=0.09822,
            # min_child_weight=7.0,
            # tree_method="hist",
            verbosity=1,
            # max_depth=3, learning_rate=0.1, n_estimators=100,
            # silent=True, objective="rank:pairwise", booster='gbtree',
            # n_jobs=-1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0,
            # subsample=1, colsample_bytree=1, colsample_bylevel=1,
            # reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
            # base_score=0.5,

        )
        # return xgboost.XGBRanker()
        # return sklearn.linear_model.Ridge()

    def _create_training_dataframe(self, users):
        """
        Creates a training dataframe with user IDs and their corresponding item recommendations.

        Parameters:
        n_users (int): The number of users.

        Returns:
        DataFrame: The training dataframe.
        """

        print("Creating training dataframe...")
        training_dataframe = pd.DataFrame(index=users, columns=["ItemID"])
        training_dataframe["UserID"] = users
        # training_dataframe.index.name = 'UserID'

        all_user_recommendations = np.zeros(
            (len(users), self.internal_cutoff_xgboost * len(self.recommenders_generate_data)), dtype=np.int64)

        for i, (_, rec_instance) in enumerate(self.recommenders_generate_data.items()):
            recs = rec_instance.recommend(users, cutoff=self.internal_cutoff_xgboost)
            all_user_recommendations[:, i * self.internal_cutoff_xgboost:(i + 1) * self.internal_cutoff_xgboost] = recs

        training_dataframe["ItemID"] = all_user_recommendations.tolist()

        return training_dataframe.explode("ItemID").drop_duplicates()

    def _populate_training_dataframe_label(self, training_dataframe):
        print("Populating training dataframe with labels...")
        urm_validation_coo = sps.coo_matrix(self.URM_val)

        correct_recommendations = pd.DataFrame({"UserID": urm_validation_coo.row,
                                                "ItemID": urm_validation_coo.col})
        training_dataframe = pd.merge(training_dataframe, correct_recommendations, on=['UserID', 'ItemID'], how='left',
                                      indicator='Exist')

        training_dataframe["Label"] = training_dataframe["Exist"] == "both"
        training_dataframe.drop(columns=['Exist'], inplace=True)
        # training_dataframe = training_dataframe.set_index('UserID')

        print("Percentage of positive samples: {:.2f}%".format(
            training_dataframe["Label"].sum() / training_dataframe.shape[0] * 100))

        return training_dataframe

    def _populate_training_dataframe(self, training_dataframe, users):
        """
        Populates the training dataframe with item scores from each recommender.

        Parameters:
        training_dataframe (DataFrame): The training dataframe.
        n_users (int): The number of users.

        Returns:
        DataFrame: The populated training dataframe.
        """
        # for user_id in tqdm.tqdm(users):
        #     for rec_label, rec_instance in self.recommenders.items():
        #         item_list = training_dataframe.loc[user_id, "ItemID"].values.tolist()
        #         all_item_scores = rec_instance._compute_item_score([user_id], items_to_compute=item_list)
        #         training_dataframe.loc[user_id, rec_label] = all_item_scores[0, item_list]

        print("Populating training dataframe with item scores...")
        for rec_label, rec_instance in tqdm.tqdm(self.recommenders.items()):
            all_user_scores = rec_instance._compute_item_score(users)  # Predict items for all users at once
            map_user_to_index = {user_id: index for index, user_id in enumerate(users)}
            user_index_pairs = training_dataframe[["UserID", "ItemID"]].values.astype(np.int64)
            users_indexes = [map_user_to_index[user_id] for user_id in user_index_pairs[:, 0]]

            training_dataframe[rec_label] = all_user_scores[users_indexes, user_index_pairs[:, 1]]

        item_popularity = np.ediff1d(sps.csc_matrix(self.URM_train).indptr)
        training_dataframe['item_popularity'] = item_popularity[training_dataframe["ItemID"].values.astype(int)]

        user_popularity = np.ediff1d(sps.csr_matrix(self.URM_train).indptr)
        training_dataframe['user_profile_len'] = user_popularity[training_dataframe["UserID"].values.astype(int)]

        training_dataframe = training_dataframe.sort_values("UserID").reset_index()
        training_dataframe.drop(columns=['index'], inplace=True)
        training_dataframe["ItemID"] = training_dataframe["ItemID"].astype('int64')
        return training_dataframe

    def fit(self, plot=False, prepare_training_df=True):
        """
        Trains the XGBoost model.
        """
        self.XGB_model = self._init_XGB_model()
        if prepare_training_df or self.training_dataframe is None:
            n_users, n_items = self.URM_train.shape
            training_dataframe = self._create_training_dataframe(range(n_users))
            training_dataframe = self._populate_training_dataframe_label(training_dataframe)
            self.training_dataframe = self._populate_training_dataframe(training_dataframe, range(n_users))

        X = self.training_dataframe.drop(columns=["Label"])
        y = self.training_dataframe["Label"]
        groups = self.training_dataframe.groupby("UserID").size().values

        # Splitting X, y into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)

        # Creating groups for train and validation sets
        groups_train = X_train.groupby("UserID").size().values
        groups_val = X_val.groupby("UserID").size().values

        # X_train = X_train.drop(columns=["UserID", "ItemID"])
        # X_val = X_val.drop(columns=["UserID", "ItemID"])

        # evals = [(X_train, y_train), (X_val, y_val)]

        # groups = training_dataframe.groupby("UserID").size().values
        # self.XGB_model.fit(X_train, y_train, group=groups, verbose=True, eval_set=[(X_test, y_test)], eval_group=groups, early_stopping_rounds=10, eval_metric=["auc","error"])
        # self.XGB_model.fit(
        #     X_train, y_train, group=groups_train, verbose=True, eval_set=evals, eval_group=[groups_train, groups_val], early_stopping_rounds=10,
        #      eval_metric=['map@10']
        # )
        # self.XGB_model.fit(
        #     X_train, y_train,
        #     # verbose=True, #eval_set=evals,
        #     # early_stopping_rounds=10,
        #     # eval_metric=['merror']
        # )
        self.XGB_model.fit(X_train, y_train, group=groups_train, verbose=True)
        y_est = self.XGB_model.predict(X_val)
        # MAP
        print("MSE: {:.2f}".format(np.mean((y_val - y_est) ** 2)))

        # self.XGB_model.fit(X_train, y_val)
        if plot:
            plot_importance(self.XGB_model, importance_type='weight', title='Weight (Frequence)')

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
        training_dataframe = self._create_training_dataframe(user_id_array)
        training_dataframe = self._populate_training_dataframe(training_dataframe, user_id_array)
        X_train = training_dataframe
        # X_train = training_dataframe.drop(columns=["UserID", "ItemID"])
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
            items_scores = np.ones(n_items) * (-100000000)
            items_scores[reranked_dataframe.loc[reranked_dataframe['UserID'] == id].ItemID.values.astype(int)] = \
                reranked_dataframe.loc[reranked_dataframe['UserID'] == id].rating_xgb.values
            result.append(items_scores)
        return np.array(result)
