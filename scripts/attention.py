from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import dynet as dy
import pickle

labels = pickle.load(open("../corpus/labels_40000.pkl"))
sentence_model_filename = "../models/sentence_structure_40k.npz"

def load_model():
    with np.load(sentence_model_filename) as data:
        sent_indices = data["sent_indices"]
        sent_mask = data["sent_mask"]
    return sent_indices, sent_mask

sent_indices, sent_mask = load_model()
sent_count_old = sent_indices.shape[0]
max_length = sent_indices.shape[1]
#print(sent_count_old, max_length)

empty_index1 = np.argwhere(np.sum(sent_mask[sent_count_old//2:sent_count_old,:], axis = 1, dtype = int)== 0)
empty_index2 = np.argwhere(np.sum(sent_mask[:sent_count_old//2,:], axis = 1, dtype = int) == 0)

l = np.union1d(empty_index1,empty_index2)
empty_count = l.shape[0]
sent_count = sent_count_old - 2*empty_count
#print(sent_count_old,sent_count)
#print(empty_count)

sent_ind = np.zeros((sent_count,max_length))
sent_masks = np.zeros((sent_count,max_length))
sent_ind[0:sent_count//2,:] = np.delete(sent_indices[:sent_count_old//2], l, axis = 0)
sent_ind[sent_count//2:sent_count,:] = np.delete(sent_indices[sent_count_old//2:sent_count_old], l, axis = 0)
sent_masks[0:sent_count//2,:] = np.delete(sent_mask[:sent_count_old//2], l, axis = 0)
sent_masks[sent_count//2:sent_count,:] = np.delete(sent_mask[sent_count_old//2:sent_count_old], l, axis = 0)
sent_masks = np.sum(sent_masks,axis=1)
sent_masks = sent_masks.astype(int)
labels = np.delete(labels, l, axis = 0)

def create_split(sent_ind,sent_masks,labels):
    #242520 train; 121260 valid; 40419 test
    trc = 242520
    vc = 121260
    tec = 40419
    train_ind = np.vstack((sent_ind[:trc,:],sent_ind[sent_count//2:sent_count//2+trc,:]))
    train_masks = np.hstack((sent_masks[:trc],sent_masks[sent_count//2:sent_count//2+trc]))
    valid_ind = np.vstack((sent_ind[trc:trc+vc,:],sent_ind[sent_count//2+trc:sent_count//2+trc+vc,:]))
    valid_masks = np.hstack((sent_masks[trc:trc+vc],sent_masks[sent_count//2+trc:sent_count//2+trc+vc]))
    test_ind = np.vstack((sent_ind[trc+vc:trc+vc+tec,:],sent_ind[sent_count//2+trc+vc:sent_count//2+trc+vc+tec,:]))
    test_masks = np.hstack((sent_masks[trc+vc:trc+vc+tec],sent_masks[sent_count//2+trc+vc:sent_count//2+trc+vc+tec]))
    return train_ind,train_masks,valid_ind,valid_masks,test_ind,test_masks

#def similarity():
    
#data is already in data numpy matrix

class decomposable_attention():
    def __init__(self,sent_indices, sent_mask, labels, embedding_dim, debug=False):
        trc = 242520
        vc = 121260
        tec = 40419
        
        self.pc = dy.Model()
        self.debug = debug
        self.embedding_dim = embedding_dim 
        self.embedding_matrix = self.pc.add_lookup_parameters((16836,embedding_dim))
        
        train,train_masks,valid,valid_masks,test,test_masks = create_split(sent_indices,sent_mask,labels)
        
        self.tr_labels = labels[:trc]
        self.v_labels = labels[trc:trc+vc]
        self.te_labels = labels[trc+vc:]
        
        self.train1 = train[:trc,:]
        self.train2 = train[trc:2*trc,:]
        
        self.valid1 = valid[:vc,:]
        self.valid2 = valid[vc:2*vc,:]
        
        self.test1 = test[:tec,:]
        self.test2 = test[tec:2*tec,:]
        
        self.tr_mask1 = train_masks[:trc]
        self.tr_mask2 = train_masks[trc:2*trc]
        
        self.v_mask1 = valid_masks[:vc]
        self.v_mask2 = valid_masks[vc:2*vc]
        
        self.te_mask1 = test_masks[:tec]
        self.te_mask2 = test_masks[tec:2*tec]
        
        self.w2 = self.pc.add_parameters((embedding_dim,2*embedding_dim))
        self.b2 = self.pc.add_parameters((embedding_dim,1))
        
        self.w3 = self.pc.add_parameters((embedding_dim,2*embedding_dim))
        self.b3 = self.pc.add_parameters((embedding_dim,1))
        
        self.w = self.pc.add_parameters((1,2*embedding_dim))
        self.b = self.pc.add_parameters((1,1))
        
        
    def forward(self,x1,x2,label,k,mode):
        
        debug = self.debug
        
        if mode=='Train':
            len1 = self.tr_mask1
            len2 = self.tr_mask2
        elif mode=='Valid':
            len1 = self.v_mask1
            len2 = self.v_mask2
        else:
            len1 = self.te_mask1
            len2 = self.te_mask2
            
        embedding_dim = self.embedding_dim
        
        w = dy.parameter(self.w)
        b = dy.parameter(self.b)
        w2 = dy.parameter(self.w2)
        b2 = dy.parameter(self.b2)
        w3 = dy.parameter(self.w3)
        b3 = dy.parameter(self.b3)
        
        embeds1 = dy.reshape(dy.lookup_batch(self.embedding_matrix,x1),(len1[k],embedding_dim))
        embeds2 = dy.reshape(dy.lookup_batch(self.embedding_matrix,x2),(len2[k],embedding_dim))

        if debug:
            print('embedding 1:', (len1[k],embedding_dim), embeds1.dim())
            print('embedding 2:', (embedding_dim,len2[k]), embeds2.dim())

        similarity = embeds1*dy.transpose(embeds2)
        if debug:
            print('similarity dimension:', (len1[k],len2[k]), similarity.dim())

        n_a = dy.softmax(similarity,d=0)
        n_b = dy.softmax(similarity,d=1)
        
        alpha = dy.transpose(n_a)*embeds1
        beta = n_b*embeds2
        
        if debug:
            print('alpha:',(len2[k],embedding_dim), alpha.dim())
            print('beta:',(len1[k],embedding_dim), beta.dim())
        
        #print((w2*dy.transpose(dy.concatenate_cols([embeds1,beta]))).npvalue().shape)
        #print(b2.npvalue().shape)
        
        v1i = w2*dy.transpose(dy.concatenate_cols([embeds1,beta])) + b2
        v2j = w3*dy.transpose(dy.concatenate_cols([embeds2,alpha])) + b3
        if debug:
            print('v1:', (embedding_dim,len1[k]), v1i.dim())
            print('v2:', (embedding_dim,len2[k]), v2j.dim())
        
        v1 = dy.mean_dim(v1i,[1],0)
        v1 = dy.reshape(v1,(embedding_dim,1)) 
        
        v2 = dy.mean_dim(v2j,[1],0)
        v2 = dy.reshape(v2,(embedding_dim,1)) 
        
        score = w*dy.concatenate([v1,v2],d=0) + b
        
        return score
        
    def train(self):
        dev_iter = 100000
        embedding_dim = self.embedding_dim
        lr = 0.0003
        trainer = dy.AdamTrainer(self.pc, alpha = lr)
        
        itr = 0
        for epochs in range(15):
            tl = 0
            for k in range(len(self.tr_labels)):
                itr += 1
                dy.renew_cg()
                x1 = self.train1[k,0:self.tr_mask1[k]]
                x2 = self.train2[k,0:self.tr_mask2[k]]
                label = self.tr_labels[k]

                score = self.forward(x1,x2,label,k,mode='Train')
                norm_score = dy.logistic(score)
                
                loss = dy.binary_log_loss(norm_score,dy.inputTensor([[label]]))
                loss.backward()
                trainer.update()
                tl += loss.scalar_value()
                #print valid scores every dev_iter iterations
#                 if itr % dev_iter == 0:
#                     self.predict('Valid', itr)
                    
            if epochs in [2,5,8,10] :
                lr /= 2
                trainer = dy.AdamTrainer(self.pc, alpha = lr)
            #print train and valid scores at the end of every epoch        
            self.predict('Train', 1+epochs, True)
            self.predict('Valid', 1+epochs, True)
        #print final scores    
        #self.predict('Train',1+epochs,True) already calculated
        #self.predict('Valid',1+epochs,True)
        self.predict('Test',1+epochs,True)
            
    def predict(self, mode, count, is_epoch_end=False):
        if is_epoch_end:
            if mode=='Valid':
                vl = 0
                preds_v = []
                for k in range(len(self.v_labels)):
                    dy.renew_cg()
                    x1 = self.valid1[k,0:self.v_mask1[k]]
                    x2 = self.valid2[k,0:self.v_mask2[k]]
                    label = self.v_labels[k]

                    score = self.forward(x1,x2,label,k,mode='Valid')
                    norm_score = dy.logistic(score)
                    preds_v.append(norm_score.scalar_value()>0.5)
                    loss = dy.binary_log_loss(norm_score,dy.inputTensor([[label]]))
                    
                    vl += loss.scalar_value()
                v_acc = sum(1 for x,y in zip(self.v_labels,preds_v) if x == y) / len(self.v_labels)
                print('Validation loss after ' + str(count) + ' epochs: ' + str(vl/len(self.v_labels)))
                print('Validation acc after ' + str(count) + ' epochs: ' + str(v_acc))

            elif mode=='Test':
                tel = 0
                preds_te = []
                for k in range(len(self.te_labels)):
                    dy.renew_cg()
                    x1 = self.test1[k,0:self.te_mask1[k]]
                    x2 = self.test2[k,0:self.te_mask2[k]]
                    label = self.te_labels[k]

                    score = self.forward(x1,x2,label,k,mode='Test')
                    norm_score = dy.logistic(score)
                    preds_te.append(norm_score.scalar_value()>0.5)
                    loss = dy.binary_log_loss(norm_score,dy.inputTensor([[label]]))

                    tel += loss.scalar_value()
                te_acc = sum(1 for x,y in zip(self.te_labels,preds_te) if x == y) / len(self.te_labels)
                print('Test loss after ' + str(count) + ' epochs: ' + str(tel/len(self.te_labels)))
                print('Test acc after ' + str(count) + ' epochs: ' + str(te_acc))

            else:
                trl = 0
                preds_tr = []
                for k in range(len(self.tr_labels)):
                    dy.renew_cg()
                    x1 = self.train1[k,0:self.tr_mask1[k]]
                    x2 = self.train2[k,0:self.tr_mask2[k]]
                    label = self.tr_labels[k]

                    score = self.forward(x1,x2,label,k,mode='Train')
                    norm_score = dy.logistic(score)
                    a = norm_score.scalar_value()>0.5
                    preds_tr.append(norm_score.scalar_value()>0.5)
                    loss = dy.binary_log_loss(norm_score,dy.inputTensor([[label]]))

                    trl += loss.scalar_value()
                tr_acc = sum(1 for x,y in zip(self.tr_labels,preds_tr) if x == y) / len(self.tr_labels)
                print('Train loss after ' + str(count) + ' epochs: ' + str(trl/len(self.tr_labels)))
                print('Train acc after ' + str(count) + ' epochs: ' + str(tr_acc))

        else:
            vl = 0
            preds_v = []
            for k in range(len(self.v_labels)):
                dy.renew_cg()
                x1 = self.valid1[k,0:self.v_mask1[k]]
                x2 = self.valid2[k,0:self.v_mask2[k]]
                label = self.v_labels[k]

                score = self.forward(x1,x2,label,k,mode='Valid')
                norm_score = dy.logistic(score)
                preds_v.append(norm_score.scalar_value()>0.5)
                loss = dy.binary_log_loss(norm_score,dy.inputTensor([[label]]))

                vl += loss.scalar_value()
            v_acc = sum(1 for x,y in zip(self.v_labels,preds_v) if x == y) / len(self.v_labels)
            print('Validation loss after ' + str(count) + ' iterations: ' + str(vl/len(self.v_labels)))
            print('Validation acc after ' + str(count) + ' iterations: ' + str(v_acc))

def main():
    
    model = decomposable_attention(sent_ind, sent_masks, labels, embedding_dim=128, debug=False)
    model.train()
    
main()   
