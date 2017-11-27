from tfkld import TFKLD
import pandas as pd
from scipy.sparse import vstack, lil_matrix
from cPickle import load, dump
from scipy.sparse.linalg import svds
import numpy, gzip
import cPickle as pickle
import os
import numpy as np
from vocab_utils import load_vocab

class DimReduction(object):
    def __init__(self, M, K):
        """ Initialization
        """
        self.M = M
        self.K = K


    def svd(self):
        U, s, Vt = svds(self.M, k=self.K)
        W = U.dot(numpy.diag(s))
        H = Vt
        return (W, H)


    def nmf(self):
        pass

data_path = "../../"
def dr(trainX, devX, testX, K=200):
    n_train = trainX.shape[0]
    n_dev = devX.shape[0]
    n_test = testX.shape[0]

    M = vstack([trainX, devX, testX])
    print 'M.shape = {}'.format(M.shape)
    dr = DimReduction(M, K)
    W, H = dr.svd()
    # Split data
    trainX = W[:n_train, :]
    devX = W[n_train:n_train+n_dev, :]
    testX = W[n_train+n_dev:, :]

    return trainX, devX, testX


def split_train(train):
    if os.path.exists(data_path + "dev_data.csv"):
        return 

    print ("Splitting train data into train and dev sets")
    df = pd.read_csv(train)
    #n = len(df)
    # Shuffle and split
    #df = df.iloc[np.random.permutation(n)]
    #per = .25
    #df[:int(per*n)].to_csv(data_path + "train_data.csv", index=False)
    #df[int(per*n):].to_csv(data_path + "dev_data.csv", index=False)

def main():
    #split_train(data_path +  "train.csv")
    
    TRAIN_DATA = data_path + "train_clean.csv"
    #DEV_DATA = data_path + "dev_data.csv"
    #TEST_DATA = data_path + "test.csv"

    print ("Loading Train data")
    train_df = pd.read_csv(TRAIN_DATA, usecols=['question1', 'question2', 'is_duplicate'])
    train_df = train_df.fillna(' ')
    trainY = train_df['is_duplicate']
    trainX = list()
    for i, row in train_df.iterrows():
        trainX.append(row['question1'])
        trainX.append(row['question2'])
    print ("Train: {0}".format(train_df.shape))
    del train_df

    tfkld = TFKLD()
    print ("Fitting the model")
    
    #tfkld.fit(trainX, trainY)
    
    vocab_file = "../vocab_embedding/vocab_quora_train.txt"
    vocab = load_vocab(vocab_file)
    #print "vocab",vocab
    tfkld._fit(trainX, trainY, vocab)
    
    pickle.dump(tfkld.weights, open("../models/tfkld_weights.pkl", "wb"))
    pickle.dump(tfkld.word2id, open("../models/tfkld_word2id.pkl", "wb"))

    #print ("Loading Dev data")
    #dev_df = pd.read_csv(DEV_DATA, usecols=['question1', 'question2'])
    #dev_df = dev_df.fillna(' ')
    #devX = list()
    #for i, row in dev_df.iterrows():
    #    devX.append(row['question1'])
    #    devX.append(row['question2'])
    #print ("Dev: {0}".format(dev_df.shape))
    #del dev_df

    #print ("Loading Test data")
    #test_df = pd.read_csv(TEST_DATA, usecols=[ 'question1', 'question2'])
    #test_df = test_df.fillna(' ')
    #testX = list()
    #for i, row in test_df.iterrows():
    #    testX.append(row['question1'])
    #    testX.append(row['question2'])
    #print ("Test: {0}".format(test_df.shape))
    #del test_df

    #print ("Train: {0}, Dev: {1}, Test: {2}".format(len(trainX), len(devX), len(testX)))

    #print ("Transforming the Train data")
    #trainX = tfkld.transform(trainX)

    #print ("Transforming the dev data")
    #devX = tfkld.transform(devX)

    #print ("Transforming the test data")
    #testX = tfkld.transform(testX)

    #trainX, devX, testX = dr(trainX, devX, testX)

    #print ("Train: {0}, Dev: {1}, Test: {2}".format(trainX.shape, devX.shape, testX.shape))

    # Save data
    #print 'Save data into file ...'

    #D = {'train_q1':trainX[::2], 'train_q2':trainX[1::2]}
    #pickle.dump(D, open(data_path + "train-tfkld-dr.pkl", "wb"))

    #D = { 'dev_q1': devX[::2], 'dev_q2': devX[1::2] }
    #pickle.dump(D, open(data_path + "dev-tfkld-dr.pkl", "wb"))

    #D = {'test_q1': testX[::2], 'test_q2': testX[1::2] }
    #pickle.dump(D, open(data_path + "test-tfkld-dr.pkl", "wb"))

    #print 'Done'

if __name__ == "__main__":
    main()
