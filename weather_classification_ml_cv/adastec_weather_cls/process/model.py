import numpy as np
import math


class DecisionStump():
    def __init__(self):
        self.polarity = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None


class Adaboost():
  
    def __init__(self, n_clf=15):
        self.n_clf = n_clf

    def fit(self, X, y):
        n_samples, n_features = np.shape(X)

        w = np.full(n_samples, (1 / n_samples))
        
        self.clfs = []

        for _ in range(self.n_clf):
            clf = DecisionStump()

            min_error = float('inf')

            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)

                for threshold in unique_values:
                    p = 1

                    prediction = np.ones(np.shape(y))

                    prediction[X[:, feature_i] < threshold] = -1

                    error = sum(w[y != prediction])
                    
                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_index = feature_i
                        min_error = error

            clf.alpha = 0.5 * math.log((1.0 - min_error) / (min_error + 1e-10))

            predictions = np.ones(np.shape(y))

            negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)

            predictions[negative_idx] = -1

            w *= np.exp(-clf.alpha * y * predictions)

            w /= np.sum(w)

            self.clfs.append(clf)

    def predict(self, X):
        n_samples = np.shape(X)[0]
        y_pred = np.zeros((n_samples, 1))

        for clf in self.clfs:
            predictions = np.ones(np.shape(y_pred))

            negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)

            predictions[negative_idx] = -1

            y_pred += clf.alpha * predictions

        y_pred = np.sign(y_pred).flatten()

        return y_pred