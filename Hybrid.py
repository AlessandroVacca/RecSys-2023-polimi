import numpy as np

from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender


# Custom Hybrid class

class HybridRecommender(BaseRecommender):
    RECOMMENDER_NAME = "Hybrid"

    def __init__(self, URM_train):
        super(HybridRecommender, self).__init__(URM_train)

    def fit(self):
        self.slim_recommender = SLIMElasticNetRecommender(self.URM_train)
        self.RP3_recommender = RP3betaRecommender(self.URM_train)
        self.slim_recommender.fit(topK=8894, l1_ratio=0.05565733019999427, alpha=0.0012979360257937668)
        self.RP3_recommender.fit(topK=101, alpha=0.3026342852596128, beta=0.058468783118329024)

    def save_model(self, folder_path, file_name=None):
        pass

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        item_weights = np.empty([len(user_id_array), 22222])
        for i in range(len(user_id_array)):

            interactions = len(self.URM_train[user_id_array[i], :].indices)

            if interactions < 4:
                w = self.RP3_recommender._compute_item_score(user_id_array[i], items_to_compute)
                item_weights[i, :] = w
            else:
                w = self.slim_recommender._compute_item_score(user_id_array[i], items_to_compute)
                item_weights[i, :] = w

        return item_weights


class LinearHybridRecommender(BaseRecommender):
    """
    This recommender merges N recommenders by weighting their ratings
    """

    RECOMMENDER_NAME = "LinearHybridRecommender"

    def __init__(self, URM_train, recommenders: list, verbose=True):
        self.RECOMMENDER_NAME = ''
        for recommender in recommenders:
            self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + recommender.RECOMMENDER_NAME[:-11]
        self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + 'HybridRecommender'

        super(LinearHybridRecommender, self).__init__(URM_train, verbose=verbose)

        self.recommenders = recommenders

    def fit(self, alphas=None):
        self.alphas = alphas

    def save_model(self, folder_path, file_name=None):
        pass

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        result = self.alphas[0] * self.recommenders[0]._compute_item_score(user_id_array, items_to_compute)
        for index in range(1, len(self.alphas)):
            result = result + self.alphas[index] * self.recommenders[index]._compute_item_score(user_id_array,
                                                                                                items_to_compute)
        return result
