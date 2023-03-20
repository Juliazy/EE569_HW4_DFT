import numpy as np
from tqdm import tqdm
from Cython_slm_loss._loss import PyLoss

def cal_entropy(prob, num_cls=10):
    prob_tmp = np.copy(prob)
    prob_tmp[prob_tmp==0] = 1
    tmp = np.sum(-1*prob*np.log(prob_tmp),axis=-1)
    return tmp/np.log(num_cls)

def class_distribution(y, num_cls=10):
    distribution = np.zeros(num_cls)
    for c in range(num_cls):
        distribution[c] = y.tolist().count(c)
    return distribution

def cal_entropy_from_y(y_array, y_distribution, num_cls):
    prob = np.zeros(num_cls)
    for c in range(num_cls):
        # prob[c] = np.sum(y_array==c)/y_distribution[c]
        prob[c] = np.sum(y_array == c) / len(y_array)
    prob = prob/np.sum(prob)

    # prob_tmp = np.copy(prob)
    # prob_tmp[prob_tmp==0] = 1
    tmp = 0
    for i in range(len(prob)):
        if prob[i] > 0:
            tmp -= prob[i] * np.log(prob[i])
    # tmp = np.sum(-1*prob*np.log(prob_tmp),axis=-1)

    return tmp/np.log(num_cls)

def cal_weighted_H(X, y, classwise, bound, num_cls=10): # weighted entropy
    # classwise = class_distribution(y, num_cls=num_cls)

    # only two bins
    if np.sum(X<bound)==0 or np.sum(X>=bound)==0:
        wH = 1
    else:
        left_y = y[X<bound]
        right_y = y[X>=bound]
        left_num = left_y.size
        right_num = right_y.size

        entropy = np.array([cal_entropy_from_y(left_y, classwise, num_cls), cal_entropy_from_y(right_y, classwise, num_cls)]).reshape(-1,1)

        num = np.array([left_num, right_num]).reshape(1,-1)
        num = num/np.sum(num, keepdims=1)

        wH = num @ entropy
    return wH

class Disc_Feature_Test():
    def __init__(self, num_class=10, num_Candidate=16, loss='entropy'):
        self.num_class = num_class
        self.B_ = num_Candidate
        self.loss = loss
        self.loss_list = []

    def binning(self, x, y, classwise):
        if np.max(x) == np.min(x):
            return 1

            # B bins (B-1) candicates of partioning point
        candidates = np.arange(np.min(x), np.max(x), (np.max(x) - np.min(x)) / (self.B_))
        candidates = candidates[1:]
        candidates = np.unique(candidates)

        loss_i = np.zeros(candidates.shape[0])

        for idx in range(candidates.shape[0]):

            loss_i[idx] = cal_weighted_H(x, y, classwise, candidates[idx], num_cls=self.num_class)

            # ''' Cython loss '''
            # loss = PyLoss()
            # loss_i[idx] = loss.calc_we(x, y, candidates[idx], x.shape[0], self.num_class)

        best_loss = np.min(loss_i)

        return best_loss

    def loss_estimation(self, x, y, classwise):
        x = x.astype('float64')
        y = y.astype('float64')
        y = y.squeeze()
        minimum_loss = self.binning(x.squeeze(), y, classwise)
        return minimum_loss

    def get_all_loss(self, X, Y):
        '''
        Parameters
        ----------
        X : shape (N, P).
        Y : shape (N).

        Returns
        -------
        feat_loss: DFT loss for all the feature dimensions. The smaller, the better.
        '''
        classwise = class_distribution(Y, num_cls=self.num_class)
        feat_loss = np.zeros(X.shape[-1])
        for k in tqdm(range(X.shape[-1])):
            feat_loss[k] = self.loss_estimation(X[:, [k]], Y, classwise)
        return feat_loss


def feature_selection(tr_X, tr_y, FStype='DFT_entropy', thrs=1.0, B=16):
    """
    This is the main function for feature selection using DFT.

    Parameters
    ----------
    tr_X : shape (N, P).
    tr_y : shape (N).
    FStype: feature selection criteria
    thrs: the percentage of kept dimension (0-1), K = thrs*P
    B: the number of bins. Default=16.

    Returns
    -------
    selected_idx: selected feature dimension index based on thrs;
    feat_score: the feature importance/ DFT loss for each of the P dimensions.
    """

    NUM_CLS = np.unique(tr_y).size
    if FStype == 'DFT_entropy':  # lower the better # more loss options will be added
        dft = Disc_Feature_Test(num_class=NUM_CLS, num_Candidate=B, loss='entropy')
        feat_score = dft.get_all_loss(tr_X, tr_y)
        feat_sorted_idx = np.argsort(feat_score)

    selected_idx = feat_sorted_idx[:int(thrs * feat_score.size)]

    return selected_idx, feat_score


if __name__ == '__main__':
    from keras.datasets import mnist
    from matplotlib import pyplot as plt
    (train_images, y_train), (test_images, y_test) = mnist.load_data()

    tr_feat = train_images.reshape(60000, -1)
    selected, dft_loss = feature_selection(tr_feat, y_train, FStype='DFT_entropy', thrs=1.0, B=16)
    plt.plot(dft_loss, label='unsorted')
    plt.plot(dft_loss[selected], label='sorted')
    plt.legend()
    plt.title('DFT_loss.png')
    plt.show()
    print('finished')



