import numpy as np
from sklearn.decomposition import TruncatedSVD
import torch


#def get_weighted_average(We, seq_loc, w, zero_loc):
def get_weighted_average(We, w):
    """
    Compute the weighted average vectors
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in sentence i
    :param w: w[i, :] are the weights for the words in sentence i
    :return: emb[i, :] are the weighted average vector for sentence i
    """
    #n_samples = seq_loc.shape[0]
    n_samples = len(w)
    #emb = np.zeros((n_samples, seq_loc.shape[1], We.shape[2]))
    #emb = torch.as_tensor(torch.from_numpy(emb), dtype = torch.float32).cuda()
    #emb = torch.zeros((n_samples, seq_loc.shape[1], We.shape[2])).cuda()
    emb = torch.zeros((n_samples, w.shape[1], We.shape[3])).cuda()
    
    for i in range(n_samples):
        #print('------- w : ', i,' -------')
        #print(w[i])
        #for j in range(seq_loc.shape[1]):
        for j in range(len(w[0])):
            count = len(torch.nonzero(w[i, j], as_tuple = False))
            if count == 0:
                emb[i, j] = We[i, j, 0]
            else:
                mul = torch.matmul(w[i,j], We[i, j])
                emb[i, j] = mul / count
            '''
            seq_list = seq_loc[i][j]
            #print('seq_list:', seq_list)

            if any(seq_list):
                for t in range(len(w[i,j])):
                    if w[i, j, t] != 0:
                        count += 1
            
            #mul = torch.matmul(w[i,j,:], We[i, seq_list, :])
            mul = torch.matmul(w[i,j], We[i, j])
            emb[i, j] = mul / count
            print(emb[i, j])
            print(ksps)
            ''' 
            isnan = torch.isnan(emb[i, j])
            if (True in isnan):
                print('--------- i:', i, ', j:', j, '---------')
                print('--------- w ------------')
                print(w[i,j,:])
                print('--------- We -----------')
                print(We[i, seq_list, :])
                print(We[i])
                print('mul:', mul)
                print('count:', count)
                print('seq_list', seq_list)
                print(osjsop)
    return emb

def compute_pc(X,npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    #print('*************** compute pc *******************')
    #with np.errstate(invalid='ignore', divide='ignore'):
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    if np.any(X == 0):
        print(X)
        print('zero')
        print(sosj)
    isnan = np.isnan(X)
    if (True in isnan):
        print(X)
        print('nan')
        print(sjo)
    isinf = np.isinf(X)
    if (True in isinf):
        print(X)
        print('inf')
        print(spjs)

    svd.fit(X)
    return svd.components_

def remove_pc(X, npc=1):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    #print('************************** remove pc **************************')
    
    X = X.cpu().detach().numpy()
    pc = compute_pc(X, npc)
    if npc==1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    XX = torch.from_numpy(XX).cuda()
    return XX


#def SIF_embedding(We, x, w, params, zero_loc):
def SIF_embedding(We, w, params):
    """
    Compute the scores between pairs of sentences using weighted average + removing the projection on the first principal component
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in the i-th sentence
    :param w: w[i, :] are the weights for the words in the i-th sentence
    :param params.rmpc: if >0, remove the projections of the sentence embeddings to their first principal component
    :return: emb, emb[i, :] is the embedding for sentence i
    """
    #print('----------- We[0] ----------------')
    #print(We[0])
    #emb = get_weighted_average(We, x, w, zero_loc)
    emb = get_weighted_average(We, w)
    #print('------------ emb[0] ---------------')
    #print(emb[0])
    #print('ori:', emb.size())
    #print('-------------------  ori -------------------')
    '''
    if  params.rmpc > 0:
        for i in range(len(emb)):
            zero_count = len(torch.nonzero(w[i], as_tuple=False))
            if zero_count != 0:
               emb[i] = remove_pc(emb[i], params.rmpc)
    '''
            #emb = remove_pc(emb, params.rmpc)
    #print('************************** SIF end *********************************')
    return emb
