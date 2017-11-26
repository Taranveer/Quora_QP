
# coding: utf-8

# In[1]:

import numpy as np
from sklearn.decomposition import TruncatedSVD
from argparse import ArgumentParser
from vocab_utils import *
#from data_utils import *
import pprint
#import phrase_tagger
from sklearn.preprocessing import normalize

def parse_args():
    parser = ArgumentParser(description= "SIF Embedding")

    # data paths
    #data_path = "../corpus-tagged/book/"
    data_path = ""
    parser.add_argument('--data_path', default=data_path, type = str)
    parser.add_argument('--sentence_file', default="sentences_train_10000.pkl",type=str)
    parser.add_argument('--embeddings_file', default='../glove.840B.300d.txt',type=str)


    #save file path
    parser.add_argument('--vocab_file', default=data_path+"vocab_quora_train.txt",type=str)
    parser.add_argument('--trimmed_embedding_file', default=data_path+"trimmed_embeddings_train_quora.npz", type=str)

    parser.add_argument('--object_file', default=data_path+"object_file_train_quora.npz", type=str)
    #parser.add_argument('--concept_filenames', default="../concepts/concepts-union-clean.txt", type=str)

    # vocab building
    parser.add_argument('--build', dest='build_vocab', action='store_true')
    parser.set_defaults(build_vocab=False)

    # remove pca component
    parser.add_argument('--no_PCA', dest='pca', action='store_false')
    parser.set_defaults(pca=True)

    parser.add_argument('--n_components', default=1, type=int)


    args = parser.parse_args()
    return args




class SIF_Model(object):
    def __init__(self, args):
        self.args = args
        self.alpha = 1e-3
        self.data = self.getData()
        self.vocab = args.vocab
        self.word_embeddings = args.word_embeddings
        self.VOCAB_SIZE = len(self.vocab)
        self.vocab_count = self.load_word_counters()
        #self.getConceptVectors()
        # self.loadModel()

    def train(self):
        self.weights = self.getWeightedProbabilities()
        self.sent_indices, self.sent_mask = self.createStructure()
        self.sent_weights = self.seq2weight(self.sent_indices, self.sent_mask, self.weights)
        print self.sent_weights.shape
        # self.saveEntries()
        self.trainEmbeddings = self.SIF_embedding(self.word_embeddings, self.sent_indices, self.sent_weights)
        print "Model Training Completed. Start Saving"

    # def saveEntries(self):
    #     np.savez_compressed(self.args.object_file, weights = self.weights,
    #                         sent_weights = self.sent_weights, sent_indices = self.sent_indices)
    #     print "done"
    #
    # def loadEntries(self):
    #     with np.load(self.args.object_file) as data:
    #         self.weights = data['weights']
    #         self.sent_indices = data['sent_indices']
    #         self.sent_weights = data['sent_weights']
    #     print "Entries Loaded"

    # def loadAndTrain(self):
    #     self.loadEntries()
    #     print "Begin Training"
    #     self.trainEmbeddings = self.SIF_embedding(self.word_embeddings, self.sent_indices, self.sent_weights)

    def saveModel(self):
        np.savez_compressed(self.args.object_file, weights = self.weights, pca_components = self.pc, sif_embeddings = self.trainEmbeddings,
                            )
        print "Model Saved"


    def loadModel(self):
        with np.load(self.args.object_file) as data:
            self.weights = data['weights']
            self.pc = data['pca_components']
            self.trainEmbeddings = data['sif_embeddings']
        print "Model Loaded"


    def getData(self):
        with open(self.args.sentence_file) as f:
            data = pickle.load(f)
            data = data[0]
            self.sentence_dict = dict(zip(range(0, len(data)), data))
            data = [x.split() for x in data]
            return data

    def load_word_counters(self):
        vocab_count = Counter()
        for line in self.data:
            vocab_count.update(line)

        # Filter Count
        filtered_dict = {k:v for k,v in vocab_count.iteritems() if k in self.vocab}
        return filtered_dict

    def getWeightedProbabilities(self):
        freqs = np.zeros((1, self.VOCAB_SIZE), dtype="float")
        for word in self.vocab:
            idx = self.vocab[word]
            freqs[0, idx] = self.vocab_count[word]

        probs = freqs / np.sum(freqs)
        weights = self.alpha / (self.alpha + probs)
        return weights

    def printShapes(self):
        print "Vocab Size: %s"%(self.VOCAB_SIZE)
        print self.weights.shape
        print self.sent_indices.shape
        print self.sent_indices.shape

    def prepare_data(self,list_of_seqs):
        lengths = [len(s) for s in list_of_seqs]
        n_samples = len(list_of_seqs)
        maxlen = np.max(lengths)
        print maxlen
        x = np.zeros((n_samples, maxlen)).astype('int32')
        x_mask = np.zeros((n_samples, maxlen)).astype('float32')
        for idx, s in enumerate(list_of_seqs):
            x[idx, :lengths[idx]] = s
            x_mask[idx, :lengths[idx]] = 1.
        x_mask = np.asarray(x_mask, dtype='float32')
        return x, x_mask



    def createStructure(self):
        sentence_indices = []
        sentence_weights = []
        for line in self.data:
            indices_list = filter(lambda x : x is not None, map(self.vocab.get, line))
            sentence_indices.append(indices_list)
        x1, m1 = self.prepare_data(sentence_indices)
        return x1, m1


    def seq2weight(self, seq, mask, weight4ind):
        weight = np.zeros(seq.shape).astype('float32')
        for i in xrange(seq.shape[0]):
            for j in xrange(seq.shape[1]):
                if mask[i, j] > 0 and seq[i, j] >= 0:
                    weight[i, j] = weight4ind[0, seq[i, j]]
        weight = np.asarray(weight, dtype='float32')
        return weight

    def get_weighted_average(self, We, x, w):
        """
        Compute the weighted average vectors
        :param We: We[i,:] is the vector for word i
        :param x: x[i, :] are the indices of the words in sentence i
        :param w: w[i, :] are the weights for the words in sentence i
        :return: emb[i, :] are the weighted average vector for sentence i
        """
        n_samples = x.shape[0]
        emb = np.zeros((n_samples, We.shape[1]))
        for i in xrange(n_samples):
            emb[i, :] = w[i, :].dot(We[x[i, :], :]) / (np.count_nonzero(w[i, :]) + 1.0)
        return emb

    def compute_pc(self,X, npc=1):
        """
        Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
        :param X: X[i,:] is a data point
        :param npc: number of principal components to remove
        :return: component_[i,:] is the i-th pc
        """
        svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
        svd.fit(X)
        return svd.components_

    def remove_pc(self, X, npc=1):
        """
        Remove the projection on the principal components
        :param X: X[i,:] is a data point
        :param npc: number of principal components to remove
        :return: XX[i, :] is the data point after removing its projection
        """
        pc = self.compute_pc(X, npc)
        self.pc = pc
        if npc == 1:
            XX = X - X.dot(pc.transpose()) * pc
        else:
            XX = X - X.dot(pc.transpose()).dot(pc)
        return XX

    def SIF_embedding(self, We, x, w):
        """
        Compute the scores between pairs of sentences using weighted average + removing the projection on the first principal component
        :param We: We[i,:] is the vector for word i
        :param x: x[i, :] are the indices of the words in the i-th sentence
        :param w: w[i, :] are the weights for the words in the i-th sentence
        :param params.rmpc: if >0, remove the projections of the sentence embeddings to their first principal component
        :return: emb, emb[i, :] is the embedding for sentence i
        """
        emb = self.get_weighted_average(We, x, w)
        #print emb.shape
        #print "Computed Embeddings"
        if self.args.pca:
            emb = self.remove_pc(emb,self.args.n_components)
        return emb

    def getSIFEmbedding(self, sent):
        corpus = [sent]
        result = corpus[0]
        words = result.split()
        indices_list = filter(lambda x: x is not None, map(self.vocab.get, words))
        
        count = len(indices_list)
        sent_embedding = np.zeros((300),dtype="float32")
        
        for idx in indices_list:
            sent_embedding = sent_embedding + self.word_embeddings[idx,:] * self.weights[0,idx]
        
        sent_embedding = sent_embedding.reshape(sent_embedding.shape[0], 1)
        sent_embedding = sent_embedding.T
        
        pc = self.pc
        #print "pc shape:", pc.shape
        #print "sent emb shape:", sent_embedding.shape
        sent_embedding = sent_embedding - sent_embedding.dot(pc.transpose())*pc
        
        if count > 0:
            sent_embedding/=count
        else:
            print "Empty Sentence"

        return sent_embedding
    
    
        
    


    def getRanking(self, sent, n=10):
        """
        Get indices of top n sentence closest to the given sentence
        param: sent - given sentence 
        param: top n - no of ranked indices
        return: ranked indices
        """
        sent_embedding = normalize(self.getSIFEmbedding(sent))
        trainEmbeddings = normalize(self.trainEmbeddings)        
        sentDistances = np.dot(trainEmbeddings, sent_embedding.T)
        sentDistances = sentDistances.reshape(sentDistances.shape[0])
        rankedIndices = sentDistances.argsort()[-n:][::-1]

        for idx in rankedIndices:
            print self.sentence_dict[idx]
        return rankedIndices
    
    def ComputePair(self, sent, index, topn=10):
        """
        Check if the given sentence's pair is in top 10 of it's similar sentences
        param: sent - given sentence
        param: index - index of given sentence, index + 10000 (index of paired sentence) ****
        param: topn - no of similar sentences
        return: rankedindices, d = {0,1} "1 if paired sentence in top 10 else 0"
        """
        rankedIndices = self.getRanking(sent, topn)
        index = 10000 + index #index of paired sentence
        if index in rankedIndices:
            return rankedIndices, 1
        else:
            return rankedIndices, 0
        
    def classifyQPairs(self, topn=10):
        """
        Check if a sentence's pair is in top 10 of it's similar sentences
        param: top n similar sentences to look at
        return: predpair tup (index, rankedidx, dup) - dup index in ranked_index or not
        """
        predPair = []
        cnt = 0.0
        for index, sent in self.sentence_dict.items():
            #print sent
            rankidx, dup = self.ComputePair(sent, index, topn)
            predPair.append((index, rankidx, dup))
            
            if cnt%200 == 0.0:
                print "progress:", cnt/10000.0
            cnt+=1
            if cnt >= 10000:
                break
        
        return predPair
    
    def pairSentenceDistance(self, sent1, sent2):
        """
        Get Sentence cosine distance between sent1 and sent2
        param: sent1, sent2
        return: sentence distance
        """
        sent1_emb = normalize(self.getSIFEmbedding(sent1))
        sent2_emb = normalize(self.getSIFEmbedding(sent2))
        #print sent1_emb.shape, sent2_emb.shape
        sent_dist = sent1_emb.dot(sent2_emb.T)
        return sent_dist
    
    def getAllSentenceDistance(self):
        """
        Get sentence distance between every pair of sentence
        return: sentence distance list
        """
        cnt = 0.0
        sent_dist_list = []
        for index1, sent1 in self.sentence_dict.items():
            index2 = 10000 + index1 #index2: paired sentence *****
            
            sent2 = self.sentence_dict[index2]
            dist = self.pairSentenceDistance(sent1, sent2)
            sent_dist_list.append(dist)
            
            if cnt%200 == 0.0:
                print "progress:", cnt/10000.0
            cnt+=1
            if cnt >= 10000:
                break
            
            
        return sent_dist_list
            
        
             
    
    def weighted_average_sim_rmpc(We,x1,x2,w1,w2, params):
        """
        Compute the scores between pairs of sentences using weighted average + removing the projection on the first principal component
        :param We: We[i,:] is the vector for word i
        :param x1: x1[i, :] are the indices of the words in the first sentence in pair i
        :param x2: x2[i, :] are the indices of the words in the second sentence in pair i
        :param w1: w1[i, :] are the weights for the words in the first sentence in pair i
        :param w2: w2[i, :] are the weights for the words in the first sentence in pair i
        :param params.rmpc: if >0, remove the projections of the sentence embeddings to their first principal component
        :return: scores, scores[i] is the matching score of the pair i
        """
        
        emb1 = self.SIF_embedding(We, x1, w1, params)
        emb2 = self.SIF_embedding(We, x2, w2, params)

        inn = (emb1 * emb2).sum(axis=1)
        emb1norm = np.sqrt((emb1 * emb1).sum(axis=1))
        emb2norm = np.sqrt((emb2 * emb2).sum(axis=1))
        scores = inn / emb1norm / emb2norm
        
        return scores
    

args = parse_args()
print pprint.pformat(args.__dict__)
if args.build_vocab:
    dataset = Dataset(args.sentence_file, args.embeddings_file)
    vocab = dataset.generate_vocab()

    #write to files
    write_vocab(vocab, args.vocab_file)
    vocab = load_vocab(args.vocab_file)
    export_trimmed_w2v_vectors(vocab, args.embeddings_file, args.trimmed_embedding_file)

args.vocab = load_vocab(args.vocab_file)
print len(args.vocab)
args.word_embeddings = get_trimmed_w2v_vectors(args.trimmed_embedding_file)
print args.word_embeddings.shape

sif_model = SIF_Model(args)
# sif_model.printShapes()


# In[ ]:



