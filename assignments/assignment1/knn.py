import numpy as np


class KNN:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):
        '''
        Uses the KNN model to predict clases for the data samples provided

        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        '''
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)

        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)

    def compute_distances_two_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run

        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            for i_train in range(num_train):
                # TODO: Fill dists[i_test][i_train]
                dist = np.sum(abs(self.train_X[i_train] - X[i_test]), axis=0)
                dists[i_test][i_train] = dist

        return dists

    def compute_distances_one_loop(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run

        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            # TODO: Fill the whole row of dists[i_test]
            # without additional loops or list comprehensions
            dist = np.sum(abs(self.train_X[:] - X[i_test]), axis=1)
            dists[i_test] = dist

        return dists

    def compute_distances_no_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run

        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        # Using float32 to to save memory - the default is float64
        dists = np.zeros((num_test, num_train), np.float32)

        # np.concatenate()
        # dist = np.sum(abs(self.train_X[:] - X[i_test]), axis=1)
        # dists = np.sum(abs(self.train_X[:, None] - X), axis=1)
        dists = (np.abs(self.train_X[:, :, None] - X.T[None, :, :]).sum(axis=1)).T

        # dists = (np.abs(self.train_X[:, :, None] - X.T[None, :, :]).sum(axis=1)).T
        return dists

    def predict_labels_binary(self, dists):
        '''
        Returns model predictions for binary classification case

        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of bool (num_test_samples) - binary predictions
           for every test sample
        '''
        # print(dists, np.shape(dists))
        # print("min", min(dists[0]), "|")
        num_test = dists.shape[0]

        pred = np.zeros(num_test, np.bool)

        for i in range(num_test):
            # TODO: Implement choosing best class based on k
            # nearest training samples


            # sorted = dists[0].sort()
            b1 = 0
            b2 = 0

            y = 0

            predk = np.zeros(self.k, np.bool) # [0,0,0]
            # pred_min_dist = dists[i][0]
            sorted = np.sort(dists[i]) # [1..12]
            for k in range(self.k): # [1..3]
                for j in range(dists[i].shape[0]): # [1..32]
                    if (sorted[k] == dists[i][j]):
                        y = self.train_y[j]
                        break
                predk[k] = (y == 0) | (y == 9)

            for k in range(self.k):
                if predk[k] == True:
                    b1 += 1
                if predk[k] == False:
                    b2 += 1

            pred[i] = b1 >= b2

            '''
            min_dist_index = 0
            min_dist = dists[i][0]
            for j in range(dists[i].shape[0]):
                dist = dists[i][j]
                if (dist < min_dist):
                    min_dist = dist
                    min_dist_index = j

            y = self.train_y[min_dist_index]
            predk[j] = (y == 0) | (y == 9)
            '''

            # pred[i] = (y == 0) | (y == 9)
            # print("predict", self.k, pred)
        return pred

    def predict_labels_multiclass(self, dists):
        '''
        Returns model predictions for multi-class classification case

        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of int (num_test_samples) - predicted class index
           for every test sample
        '''

        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.int)
        for i in range(num_test):
            # TODO: Implement choosing best class based on k
            # nearest training samples


            # sorted = dists[0].sort()
            b1 = 0
            b2 = 0

            y = 0

            predk = {} # [0,0,0]
            num_of_digits = np.zeros(10, np.int)
            # pred_min_dist = dists[i][0]
            sorted = np.sort(dists[i]) # [1..12]
            for k in range(self.k): # [1..3]
                for j in range(dists[i].shape[0]): # [1..32]
                    if (sorted[k] == dists[i][j]):
                        y = self.train_y[j]
                        break
                predk[k] = [y, 0]

            for k in range(self.k):
                index = predk[k]
                num_of_digits[index] = num_of_digits[index] + 1

            max_index = 0
            max = num_of_digits[max_index]
            for j in range(num_of_digits.shape[0]):
                if num_of_digits[j] > max:
                    max = num_of_digits[j]
                    max_index = j

            pred[i] = max_index

            '''
            min_dist_index = 0
            min_dist = dists[i][0]
            for j in range(dists[i].shape[0]):
                dist = dists[i][j]
                if (dist < min_dist):
                    min_dist = dist
                    min_dist_index = j

            y = self.train_y[min_dist_index]
            predk[j] = (y == 0) | (y == 9)
            '''

            # pred[i] = (y == 0) | (y == 9)
            # print("predict", self.k, pred)
        return pred
