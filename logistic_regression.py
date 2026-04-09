# CS542 Fall 2025 Programming Assignment 1
# Logistic Regression Classifier

import os
import numpy as np
from collections import defaultdict
from math import ceil
from random import Random


'''
Computes the logistic function.
'''
def sigma(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegression():

    def __init__(self, n_features=4):
        
        # Be sure to use the right class_dict for each data set
        self.class_dict = {'neg': 0, 'pos': 1}
#         self.class_dict = {'action': 0, 'comedy': 1}

        # Use of self.feature_dict is optional for this assignment
        # self.feature_dict = {'fast': 0, 'couple': 1, 'shoot': 2, 'fly': 3, 'fun':4} # comment out
        
        # Use with self.feature_dict:
        self.n_features = len(self.feature_dict) #comment out

        # Use with self.class_dict (when self.feature_dict is not used):
        self.n_features = n_features 
    
        self.theta = np.zeros(self.n_features + 1) # weights (and bias)

    '''
    Loads a dataset. Specifically, returns a list of filenames, and dictionaries
    of classes and documents such that:
    classes[filename] = class of the document
    documents[filename] = feature vector for the document (use self.featurize)
    '''
    def load_data(self, data_set):
        filenames = []
        classes = dict()
        documents = dict()
        # iterate over documents
        for root, dirs, files in os.walk(data_set):
            label = os.path.basename(root) # Extracts the last component of the path
            if label not in self.class_dict:
                continue 
            for name in files:
                path = os.path.join(root, name)
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    # BEGIN STUDENT CODE
                    # Store list of tokens. We featurize in self.featurize(documents[name])
                    # grader-safe
                    # END STUDENT CODE
                    text = f.read().strip()
                    tokens = text.lower().split()  # simple tokenization
                # store info
                    filenames.append(name)
                    classes[name] = self.class_dict[label]
                    documents[name] = self.featurize(tokens)
        return filenames, classes, documents

    '''
    Given a document (as a list of words), returns a feature vector.
    Note that the last element of the vector, corresponding to the bias, is a
    "dummy feature" with value 1.
    '''
   
    
    def featurize(self, document):
        vector = np.zeros(self.n_features + 1)
        
        # Rail guard. Works in case documents are lists or strings.
        tokens = document if isinstance(document, list) else str(document).lower().split() 
        
        # your code here
        # BEGIN STUDENT CODE
        for token in tokens:
            if token in self.feature_dict:           # only keep words in vocabulary
                idx = self.feature_dict[token]       # get index of this token
                vector[idx] += 1   
        # END STUDENT CODE
        
        vector[-1] = 1
        return vector

    '''
    Trains a logistic regression classifier on a training set.
    '''
    def train(self, train_set, batch_size=3, n_epochs=1, eta=0.1):
        filenames, classes, documents = self.load_data(train_set) 
        filenames = sorted(filenames)
        n_minibatches = ceil(len(filenames) / batch_size)
        
        for epoch in range(n_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
            loss = 0
            
            for i in range(n_minibatches):
                # list of filenames in minibatch
                minibatch = filenames[i * batch_size: (i + 1) * batch_size]
                
                # Explore first document and get info about the shape (to make sure x and theta are compatible)
                first_vector = self.featurize(documents[minibatch[0]])
                shape = first_vector.shape[0]
                
                # Preventing theta to be of wrong length
                if getattr(self, "theta", None) is None or self.theta.shape[0] != shape:
                    self.theta = np.zeros(shape, dtype=np.float64)
                    
                # create and fill in matrix x and vector y
                # BEGIN STUDENT CODE
                            # create and fill in matrix X and vector y
                    X = np.array([self.featurize(documents[f]) for f in minibatch])
                    y = np.array([classes[f] for f in minibatch])

                    # compute y_hat (predicted probabilities)
                    z = X @ self.theta
                    y_hat = sigma(z)    # sigmoid already implemented

                    # update loss (cross-entropy)
                    # add a small epsilon to avoid log(0)
                    eps = 1e-15
                    batch_loss = -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))
                    loss += batch_loss * len(minibatch)   # scale by batch size

                    # compute gradient
                    grad = (X.T @ (y_hat - y)) / len(minibatch)

                    # update weights (and bias is included since last column in X = 1)
                    self.theta -= eta * grad

                
                        # compute y_hat
                        # update loss
                        # Prevent logs from seeing any zeros (avoid getting infinites during training)
                        # compute gradient
                        # update weights (and bias)
                        # END STUDENT CODE
                
            loss /= len(filenames)
            print("Average Train Loss: {}".format(loss))
            # randomize order
            Random(epoch).shuffle(filenames)
            filenames.sort()

    '''
    Tests the classifier on a development or test set.
    Returns a dictionary of filenames mapped to their correct and predicted
    classes such that:
    results[filename]['correct'] = correct class
    results[filename]['predicted'] = predicted class
    '''
    def test(self, dev_set):
        results = defaultdict(dict)
        filenames, classes, documents = self.load_data(dev_set)
        for name in filenames:
            # get most likely class (recall that P(y=1|x) = y_hat)
            # BEGIN STUDENT CODE
            # get feature vector for this document
            x = documents[name]

            # compute probability that y=1 (positive review)
            y_hat = sigma(x @ self.theta)

            # apply decision boundary at 0.5
            predicted = 1 if y_hat > 0.5 else 0

            # store results
            results[name]['correct'] = classes[name]      # true label
            results[name]['predicted'] = predicted        # model prediction
                # END STUDENT CODE
        return results

    '''
    Given results, calculates the following:
    Precision, Recall, F1 for each class
    Accuracy overall
    Also, prints evaluation metrics in readable format.
    '''

    # def evaluate(self, results):
    #     pass
    
    def evaluate(self, results):
        # Initialize counters
        TP = FP = TN = FN = 0
        total = len(results)
        correct = 0

        # Loop over results
        for name, r in results.items():
            y_true = r['correct']
            y_pred = r['predicted']

            if y_true == y_pred:
                correct += 1

            if y_true == 1 and y_pred == 1:
                TP += 1
            elif y_true == 0 and y_pred == 1:
                FP += 1
            elif y_true == 0 and y_pred == 0:
                TN += 1
            elif y_true == 1 and y_pred == 0:
                FN += 1

        # Compute metrics (add epsilon to avoid divide by zero)
        eps = 1e-15
        accuracy = correct / total
        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        # Print results
        print("Evaluation Results")
        print("===================")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }



if __name__ == '__main__':
    
    # Use with self.feature_dict:
    # lr = LogisticRegression() 

#     Use with self.class_dict (when self.feature_dict is not used):
    lr = LogisticRegression(n_features=400) 
    
    # Make sure these point to the right directories
    lr.train('movie_reviews/train', batch_size=300, n_epochs=100, eta=0.01)
#     lr.train('movie_reviews_small/train', batch_size=1, n_epochs=10, eta=0.1)
#     lr.train('movie_reviews_test/train', batch_size=1, n_epochs=10, eta=0.1)
    results = lr.test('movie_reviews/dev')
#     results = lr.test('movie_reviews_small/test')
#     results = lr.test('movie_reviews_test/test')
    print("results size:", len(results))
    lr.evaluate(results)
