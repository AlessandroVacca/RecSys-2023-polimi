import numpy as np

from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import *

from numpy import linalg as LA
import scipy.sparse as sps


# Custom Hybrid class

class HybridRecommender(BaseRecommender):
    RECOMMENDER_NAME = "Hybrid"

    def __init__(self, URM_train):
        super(HybridRecommender, self).__init__(URM_train)

    def fit(self):
        self.slim_recommender = MultiThreadSLIM_SLIMElasticNetRecommender(self.URM_train)
        self.RP3_recommender = RP3betaRecommender(self.URM_train)
        self.userknn_recommender = UserKNNCFRecommender(self.URM_train)
        self.slim_recommender.fit(topK=8894, l1_ratio=0.05565733019999427, alpha=0.0012979360257937668, workers = 7)
        self.RP3_recommender.fit(topK=101, alpha=0.3026342852596128, beta=0.058468783118329024)
        self.userknn_recommender.fit(topK=469, shrink=38, similarity='asymmetric', normalize=True,
                                       feature_weighting='TF-IDF', asymmetric_alpha=0.40077406933762383)

    def save_model(self, folder_path, file_name=None):
        pass

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        item_weights = np.empty([len(user_id_array), 22222])
        for i in range(len(user_id_array)):

            interactions = len(self.URM_train[user_id_array[i], :].indices)

            if interactions == 1:
                w = self.userknn_recommender._compute_item_score(user_id_array[i], items_to_compute)
                item_weights[i, :] = w
            # elif interactions == 2: or interactions == 3:
            #     w = self.RP3_recommender._compute_item_score(user_id_array[i], items_to_compute)
            #     item_weights[i, :] = w
            if interactions >= 2 & interactions <= 250:
                w = self.slim_recommender._compute_item_score(user_id_array[i], items_to_compute)
                item_weights[i, :] = w
            else:
                w = self.RP3_recommender._compute_item_score(user_id_array[i], items_to_compute)
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


class LinearZScoreNormalizedHybridRecommender(BaseRecommender):
    RECOMMENDER_NAME = "LinearZScoreNormalizedHybridRecommender"

    def __init__(self, URM_train, recommenders: list, verbose=True):
        self.RECOMMENDER_NAME = ''
        for recommender in recommenders:
            self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + recommender.RECOMMENDER_NAME[:-11]
        self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + 'ZScoreNormHybridRecommender'

        super(LinearZScoreNormalizedHybridRecommender, self).__init__(URM_train, verbose=verbose)

        self.recommenders = recommenders

    def fit(self, alphas=None):
        self.alphas = alphas
        self.mean_predictions = []
        self.std_predictions = []

        # Fit each individual recommender
        for recommender in self.recommenders:
            # Compute mean and standard deviation of predictions
            predictions = recommender._compute_item_score(np.arange(self.URM_train.shape[0]))
            self.mean_predictions.append(np.mean(predictions))
            self.std_predictions.append(np.std(predictions))

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        result = self.alphas[0] * self.normalize_predictions(self.recommenders[0]._compute_item_score(user_id_array, items_to_compute), 0)
        for index in range(1, len(self.alphas)):
            result = result + self.alphas[index] * self.normalize_predictions(self.recommenders[index]._compute_item_score(user_id_array, items_to_compute), index)
        return result

    def normalize_predictions(self, predictions, index):
        # Apply z-score normalization using mean and std of predictions from each recommender
        return (predictions - self.mean_predictions[index]) / self.std_predictions[index]

class DifferentLossScoresHybridRecommender(BaseRecommender):
    """ ScoresHybridRecommender
    Hybrid of two prediction scores R = R1/norm*alpha + R2/norm*(1-alpha) where R1 and R2 come from
    algorithms trained on different loss functions.

    """

    RECOMMENDER_NAME = "DifferentLossScoresHybridRecommender"

    def __init__(self, URM_train, recommender_1, recommender_2):
        super(DifferentLossScoresHybridRecommender, self).__init__(URM_train)

        self.URM_train = sps.csr_matrix(URM_train)
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2

    def fit(self, norm, alpha=0.5):

        self.alpha = alpha
        self.norm = norm

    def _compute_item_score(self, user_id_array, items_to_compute):

        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)

        norm_item_weights_1 = LA.norm(item_weights_1, self.norm)
        norm_item_weights_2 = LA.norm(item_weights_2, self.norm)

        if norm_item_weights_1 == 0:
            raise ValueError(
                "Norm {} of item weights for recommender 1 is zero. Avoiding division by zero".format(self.norm))

        if norm_item_weights_2 == 0:
            raise ValueError(
                "Norm {} of item weights for recommender 2 is zero. Avoiding division by zero".format(self.norm))

        item_weights = item_weights_1 / norm_item_weights_1 * self.alpha + item_weights_2 / norm_item_weights_2 * (
                1 - self.alpha)

        return item_weights


class MajVotHybridRecommender(BaseRecommender):
    RECOMMENDER_NAME = "MajorityVotingHybrid"

    def __init__(self, URM_train, recommenders):
        super(MajVotHybridRecommender, self).__init__(URM_train)
        self.recommenders = recommenders

    def fit(self):
        # Fit each individual recommender
        for recommender in self.recommenders:
            recommender.fit()

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        Computes the score for each item for the given users.

        :param user_id_array: array containing the user indices whose recommendations need to be computed
        :param items_to_compute: array containing the items whose scores are to be computed.
                                 If None, all items are computed.
        :return: array (len(user_id_array), n_items) with the score.
        """
        # Initialize an array to store the aggregated votes
        aggregated_votes = None

        # Iterate over each recommender
        for recommender in self.recommenders:
            # Compute scores for each recommender
            scores = recommender._compute_item_score(user_id_array, items_to_compute)

            # Initialize the aggregated_votes array with the shape of scores from the first recommender
            if aggregated_votes is None:
                aggregated_votes = np.zeros_like(scores)

            # Convert scores to binary votes (1 for recommended items, 0 otherwise)
            recommender_votes = scores > 0
            aggregated_votes += recommender_votes

        return aggregated_votes

    # Other methods of the class
