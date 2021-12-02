import argparse
import scipy.sparse as sp
import numpy as np
import torch
import math
import pdb
from scipy.io import loadmat
import networkx as nx
import multiprocessing as mp
import torch.nn.functional as F
from functools import partial
import random
from sklearn.metrics import roc_auc_score, f1_score
from copy import deepcopy
from scipy.spatial.distance import pdist,squareform
from torch_sparse import SparseTensor, coalesce

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--nhid', type=int, default=64)
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--size', type=int, default=100)

    parser.add_argument('--epochs', type=int, default=2010,
                help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--batch_nums', type=int, default=6000, help='number of batches per epoch')
    parser.add_argument('--batch_size', type=int, default=40, help='number of batches per epoch')


    parser.add_argument('--imbalance', action='store_true', default=False)
    parser.add_argument('--setting', type=str, default='no', 
        choices=['no','upsampling', 'smote','reweight','embed_up', 'recon','newG_cls','recon_newG'])
    #upsampling: oversample in the raw input; smote: ; reweight: reweight minority classes; 
    # embed_up: 
    # recon: pretrain; newG_cls: pretrained decoder; recon_newG: also finetune the decoder

    parser.add_argument('--opt_new_G', action='store_true', default=False) # whether optimize the decoded graph based on classification result.
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--up_scale', type=float, default=1)
    parser.add_argument('--im_ratio', type=float, default=0.5)
    parser.add_argument('--rec_weight', type=float, default=0.000001)
    parser.add_argument('--model', type=str, default='sage', 
        choices=['sage','gcn','GAT'])



    return parser

def split_arti(labels, c_train_num):
    #labels: n-dim Longtensor, each element in [0,...,m-1].
    #cora: m=7
    num_classes = len(set(labels.tolist()))
    c_idxs = [] # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)
    c_num_mat[:,1] = 25
    c_num_mat[:,2] = 55

    for i in range(num_classes):
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        print('{:d}-th class sample number: {:d}'.format(i,len(c_idx)))
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        train_idx = train_idx + c_idx[:c_train_num[i]]
        c_num_mat[i,0] = c_train_num[i]

        val_idx = val_idx + c_idx[c_train_num[i]:c_train_num[i]+25]
        test_idx = test_idx + c_idx[c_train_num[i]+25:c_train_num[i]+80]

    random.shuffle(train_idx)

    #ipdb.set_trace()

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)
    #c_num_mat = torch.LongTensor(c_num_mat)


    return train_idx, val_idx, test_idx, c_num_mat

def split_genuine(labels):
    #labels: n-dim Longtensor, each element in [0,...,m-1].
    #cora: m=7
    num_classes = len(set(labels.tolist()))
    c_idxs = [] # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)

    for i in range(num_classes):
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        c_num = len(c_idx)
        print('{:d}-th class sample number: {:d}'.format(i,len(c_idx)))
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        if c_num <4:
            if c_num < 3:
                print("too small class type")
                ipdb.set_trace()
            c_num_mat[i,0] = 1
            c_num_mat[i,1] = 1
            c_num_mat[i,2] = 1
        else:
            c_num_mat[i,0] = int(c_num/4)
            c_num_mat[i,1] = int(c_num/4)
            c_num_mat[i,2] = int(c_num/2)


        train_idx = train_idx + c_idx[:c_num_mat[i,0]]

        val_idx = val_idx + c_idx[c_num_mat[i,0]:c_num_mat[i,0]+c_num_mat[i,1]]
        test_idx = test_idx + c_idx[c_num_mat[i,0]+c_num_mat[i,1]:c_num_mat[i,0]+c_num_mat[i,1]+c_num_mat[i,2]]

    random.shuffle(train_idx)

    #ipdb.set_trace()

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)
    #c_num_mat = torch.LongTensor(c_num_mat)


    return train_idx, val_idx, test_idx, c_num_mat


def print_edges_num(dense_adj, labels):
    c_num = labels.max().item()+1
    dense_adj = np.array(dense_adj)
    labels = np.array(labels)

    for i in range(c_num):
        for j in range(c_num):
            #ipdb.set_trace()
            row_ind = labels == i
            col_ind = labels == j

            edge_num = dense_adj[row_ind].transpose()[col_ind].sum()
            print("edges between class {:d} and class {:d}: {:f}".format(i,j,edge_num))


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def print_class_acc(output, labels, class_num_list, pre='valid'):
    pre_num = 0
    #print class-wise performance
    '''
    for i in range(labels.max()+1):
        
        cur_tpr = accuracy(output[pre_num:pre_num+class_num_list[i]], labels[pre_num:pre_num+class_num_list[i]])
        print(str(pre)+" class {:d} True Positive Rate: {:.3f}".format(i,cur_tpr.item()))

        index_negative = labels != i
        labels_negative = labels.new(labels.shape).fill_(i)
        
        cur_fpr = accuracy(output[index_negative,:], labels_negative[index_negative])
        print(str(pre)+" class {:d} False Positive Rate: {:.3f}".format(i,cur_fpr.item()))

        pre_num = pre_num + class_num_list[i]
    '''

    #ipdb.set_trace()
    if labels.max() > 1:
        auc_score = roc_auc_score(labels.detach(), F.softmax(output, dim=-1).detach(), average='macro', multi_class='ovr')
    else:
        auc_score = roc_auc_score(labels.detach(), F.softmax(output, dim=-1)[:,1].detach(), average='macro')

    macro_F = f1_score(labels.detach(), torch.argmax(output, dim=-1).detach(), average='macro')
    print(str(pre)+' current auc-roc score: {:f}, current macro_F score: {:f}'.format(auc_score,macro_F))

    return

def src_upsample(adj,features,labels,idx_train, portion=1.0, im_class_num=3):
    c_largest = labels.max().item()
    adj_back = adj.to_dense()
    chosen = None

    #ipdb.set_trace()
    avg_number = int(idx_train.shape[0]/(c_largest+1))

    for i in range(im_class_num):
        new_chosen = idx_train[(labels==(c_largest-i))[idx_train]]
        if portion == 0:#refers to even distribution
            c_portion = int(avg_number/new_chosen.shape[0])

            for j in range(c_portion):
                if chosen is None:
                    chosen = new_chosen
                else:
                    chosen = torch.cat((chosen, new_chosen), 0)

        else:
            c_portion = int(portion)
            portion_rest = portion-c_portion
            for j in range(c_portion):
                num = int(new_chosen.shape[0])
                new_chosen = new_chosen[:num]

                if chosen is None:
                    chosen = new_chosen
                else:
                    chosen = torch.cat((chosen, new_chosen), 0)
            
            num = int(new_chosen.shape[0]*portion_rest)
            new_chosen = new_chosen[:num]

            if chosen is None:
                chosen = new_chosen
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
            

    add_num = chosen.shape[0]
    new_adj = adj_back.new(torch.Size((adj_back.shape[0]+add_num, adj_back.shape[0]+add_num)))
    new_adj[:adj_back.shape[0], :adj_back.shape[0]] = adj_back[:,:]
    new_adj[adj_back.shape[0]:, :adj_back.shape[0]] = adj_back[chosen,:]
    new_adj[:adj_back.shape[0], adj_back.shape[0]:] = adj_back[:,chosen]
    new_adj[adj_back.shape[0]:, adj_back.shape[0]:] = adj_back[chosen,:][:,chosen]

    #ipdb.set_trace()
    features_append = deepcopy(features[chosen,:])
    labels_append = deepcopy(labels[chosen])
    idx_new = np.arange(adj_back.shape[0], adj_back.shape[0]+add_num)
    idx_train_append = idx_train.new(idx_new)

    features = torch.cat((features,features_append), 0)
    labels = torch.cat((labels,labels_append), 0)
    idx_train = torch.cat((idx_train,idx_train_append), 0)
    adj = new_adj.to_sparse()

    return adj, features, labels, idx_train



def src_smote(adj,features,labels,idx_train, portion=1.0, im_class_num=3):
    c_largest = labels.max().item()
    adj_back = adj.to_dense()
    chosen = None
    new_features = None

    #ipdb.set_trace()
    avg_number = int(idx_train.shape[0]/(c_largest+1))

    for i in range(im_class_num):
        new_chosen = idx_train[(labels==(c_largest-i))[idx_train]]
        if portion == 0:#refers to even distribution
            c_portion = int(avg_number/new_chosen.shape[0])

            portion_rest = (avg_number/new_chosen.shape[0]) - c_portion

        else:
            c_portion = int(portion)
            portion_rest = portion-c_portion
            
        for j in range(c_portion):
            num = int(new_chosen.shape[0])
            new_chosen = new_chosen[:num]

            chosen_embed = features[new_chosen,:]
            distance = squareform(pdist(chosen_embed.detach()))
            np.fill_diagonal(distance,distance.max()+100)

            idx_neighbor = distance.argmin(axis=-1)
            
            interp_place = random.random()
            embed = chosen_embed + (chosen_embed[idx_neighbor,:]-chosen_embed)*interp_place

            if chosen is None:
                chosen = new_chosen
                new_features = embed
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
                new_features = torch.cat((new_features, embed),0)
            
        num = int(new_chosen.shape[0]*portion_rest)
        new_chosen = new_chosen[:num]

        chosen_embed = features[new_chosen,:]
        distance = squareform(pdist(chosen_embed.detach()))
        np.fill_diagonal(distance,distance.max()+100)

        idx_neighbor = distance.argmin(axis=-1)
            
        interp_place = random.random()
        embed = chosen_embed + (chosen_embed[idx_neighbor,:]-chosen_embed)*interp_place

        if chosen is None:
            chosen = new_chosen
            new_features = embed
        else:
            chosen = torch.cat((chosen, new_chosen), 0)
            new_features = torch.cat((new_features, embed),0)
            

    add_num = chosen.shape[0]
    new_adj = adj_back.new(torch.Size((adj_back.shape[0]+add_num, adj_back.shape[0]+add_num)))
    new_adj[:adj_back.shape[0], :adj_back.shape[0]] = adj_back[:,:]
    new_adj[adj_back.shape[0]:, :adj_back.shape[0]] = adj_back[chosen,:]
    new_adj[:adj_back.shape[0], adj_back.shape[0]:] = adj_back[:,chosen]
    new_adj[adj_back.shape[0]:, adj_back.shape[0]:] = adj_back[chosen,:][:,chosen]

    #ipdb.set_trace()
    features_append = deepcopy(new_features)
    labels_append = deepcopy(labels[chosen])
    idx_new = np.arange(adj_back.shape[0], adj_back.shape[0]+add_num)
    idx_train_append = idx_train.new(idx_new)

    features = torch.cat((features,features_append), 0)
    labels = torch.cat((labels,labels_append), 0)
    idx_train = torch.cat((idx_train,idx_train_append), 0)
    adj = new_adj.to_sparse()

    return adj, features, labels, idx_train

def recon_upsample(embed, labels, idx_train, adj=None, portion=1.0, im_class_num=3):
    '''
    Perform upsampling on a batch of embeddings
    Parameters:
    embed (Tensor): M x N tensor with M samples and N features
    labels (Tensor): 1D tensor with M integer labels
    idx_train (Tensor): 1D tensor with an indexed list of samples in the training set
    adj (SparseTensor): 2D M X M tensor with adjacencies
    portion (float): 
    im_class_num (int): imbalanced class label
    Returns:
    new_embed (Tensor): embeddings tensor with original and augmented nodes
    labels (Tensor): labels for original and augmented nodes
    idx_train (Tensor): Indexed list of samples in the training set (including augmented nodes)
    new_adj (SparseTensor): New adjacency matrix (with edges for augmented nodes.)
    '''

    c_largest = labels.max().item()
    avg_number = int(idx_train.shape[0]/(c_largest+1))
    #ipdb.set_trace()
    adj_new = None

    # helper function
    def upsample_embeddings(embed, idx_curr_class_interp, take_from_other_classes=False):
        if take_from_other_classes:
            # there's no other samples in this class, so we'll find the closest sample to interpolate

            # samples of the OTHER classes
            idx_other_class = idx_train[(labels!=(c_largest-i))[idx_train]]
            chosen_embed = torch.cat((
                    embed[idx_other_class],
                    embed[idx_curr_class_interp],
                ), dim=0
            )
            distance = to_square(torch.nn.functional.pdist(chosen_embed.detach()))
            distance.fill_diagonal_(distance.max()+100)
            # we're interested only in the distance to the positive sample
            idx_neighbor = distance[-1].argmin(axis=-1)
            # absolute index of the closest neighbor
            idx_neighbor_abs = idx_other_class[idx_neighbor]
        else:
            if len(idx_curr_class_interp)>0:
                chosen_embed = embed[idx_curr_class_interp]
                distance = to_square(torch.nn.functional.pdist(chosen_embed.detach()))
                distance.fill_diagonal_(distance.max()+100)
                idx_neighbor = distance.argmin(axis=-1)
                # absolute index of the closest neighbor
                idx_neighbor_abs = idx_curr_class_interp[idx_neighbor]
            else:
                # we can't interpolate because there's no sample of the target class in this batch
                return embed[idx_curr_class_interp,:], []

        interp_place = random.random()
        return embed[idx_curr_class_interp,:] + (chosen_embed[idx_neighbor,:]-embed[idx_curr_class_interp,:])*interp_place, idx_neighbor_abs

    # start upsampling
    for i in range(im_class_num):
        # samples of the current class
        idx_curr_class = idx_train[(labels==(c_largest-i))[idx_train]]
        # determine # of samples to be used for new interpolated samples
        num = int(idx_curr_class.shape[0]*portion)
        
        take_from_other_classes = False
        if idx_curr_class.shape[0]==1:
            # if there's only 1 sample, we have to do things differently
            take_from_other_classes = True
            num = 1

        if portion == 0:
            c_portion = int(avg_number/idx_curr_class.shape[0])
            num = idx_curr_class.shape[0]
        else:
            c_portion = 1

        for j in range(c_portion):
            # samples of current class for interpolation
            idx_curr_class_interp = idx_curr_class[:num]

            new_embed, idx_neighbor_abs = upsample_embeddings(embed, idx_curr_class_interp, take_from_other_classes=take_from_other_classes)

            new_labels = labels.new(torch.Size((idx_curr_class_interp.shape[0],1))).reshape(-1).fill_(c_largest-i)
            idx_new = np.arange(embed.shape[0], embed.shape[0]+idx_curr_class_interp.shape[0])
            idx_train_append = idx_train.new(idx_new)

            embed = torch.cat((embed,new_embed), 0)
            labels = torch.cat((labels,new_labels), 0)
            idx_train = torch.cat((idx_train,idx_train_append), 0)

            if adj is not None:
                # connections from the original nodes, plus connections of the neighbor will be added to the new nodes
                if adj_new is None:
                    adj_new = combine_sparse_adj(adj, idx_curr_class_interp, idx_neighbor_abs)
                else:
                    adj_new = combine_sparse_adj(adj_new, idx_curr_class_interp, idx_neighbor_abs)

    if adj is not None:
        return embed, labels, idx_train, adj_new.detach()

    else:
        return embed, labels, idx_train

def adj_mse_loss(adj_rec, adj_tgt, adj_mask = None):
    edge_num = adj_tgt.nonzero().shape[0]
    total_num = adj_tgt.shape[0]**2

    neg_weight = edge_num / ((total_num-edge_num)+ 1e-9)

    weight_matrix = adj_rec.new(adj_tgt.shape).fill_(1.0)
    weight_matrix[adj_tgt==0] = neg_weight

    loss = torch.sum(weight_matrix * (adj_rec - adj_tgt) ** 2)

    return loss


def to_square(a):
    '''
    Makeshift equivalent to scipy's squareform
    '''

    s = list(a.shape)
    assert len(s)==1,"Argument is not a distance vector."

    # from scipy's squareform:
    # Grab the closest value to the square root of the number
    # of elements times 2 to see if the number of elements
    # is indeed a binomial coefficient.
    d = int(math.ceil(math.sqrt(s[0] * 2)))

    # Check that v is of valid dimensions.
    if d * (d - 1) != s[0] * 2:
        raise ValueError('Incompatible vector size. It must be a binomial '
                            'coefficient n choose 2 for some integer n >= 2.')

    b = torch.zeros((d,d), dtype=a.dtype)
    p2=d-1
    p1=0
    for i in range(d):
        b[i,i+1:] = a[p1:p1+p2]
        p1+=p2
        p2-=1
    return b + b.T


# based on https://github.com/rusty1s/pytorch_sparse/issues/113
def add_sparse(a, b, clamp=True, preserve_size=True):
    # assert a.sizes() == b.sizes(), "The Tensor dimensions do not match"
    row_a, col_a, values_a = a.coo()
    row_b, col_b, values_b = b.coo()
    
    index = torch.stack([torch.cat([row_a, row_b]), torch.cat([col_a, col_b])])
    value = torch.cat([values_a, values_b])
    
    if not preserve_size:
        m,n = a.sizes()
    else:
        # we're only interested in square adj matrices
        m = n = index.max().item()+1
    index, value = coalesce(index, value, m=m, n=n)
    if clamp:
        value = torch.clamp(value, 0, 1)
    res = SparseTensor.from_edge_index(index, value, sparse_sizes=(m, n))
    return res


def combine_sparse_adj(adj, subset1, subset2):
    """
    Combines the adjacencies of two subsets of nodes (specifically, the original nodes selected to interpolate, and their closest neighbor nodes).
    The combined adjacencies are appended back (as rows/cols) to the original adjacency matrix.
    Arguments:
    adj (SparseTensor): sparse adjacency matrix
    subset1 (Tensor): list of indices (rows in adj) belonging to the 1st subset
    subset2 (Tensor): list of indices (rows in adj) belonging to the 2nd subset
    Returns: 
    mod_adj (SparseTensor): a modified adjacency matrix combining adjacencies of subset1 and subset2

    Example - Combine rows 0 and 2 of SparseTensor x, appending them back as additional rows and cols:

    a = torch.tensor([[0, 1, 2],[2, 1, 0]], dtype=torch.long)
    b = torch.ones((3), dtype=torch.long)
    x = SparseTensor.from_edge_index(
        edge_index=a,
        edge_attr=b
    )
    print(x.to_dense())
    =>
    tensor([[0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]])

    combine_sparse_adj(x, 0, 2).to_dense()
    =>
    tensor([[0, 0, 1, 1],
            [0, 1, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 0]])

    """
    offset_rows, offset_cols = adj.sparse_sizes()
    
    combined = add_sparse(adj[subset1,:], adj[subset2,:], preserve_size=False)
    rows_to_add = combined.sizes()[0]

    # shift rows 
    r, c, v = combined.coo()
    r = r + offset_rows
    
    newadj = SparseTensor.from_edge_index(
        edge_index=torch.stack((r, c), 0),
        edge_attr=v,
        sparse_sizes=[s+rows_to_add for s in adj.sizes()]
    )

    # shift cols
    r, c, v = combined.t().coo()
    c = c + offset_cols

    combined_t = SparseTensor.from_edge_index(
        edge_index=torch.stack((r,c), 0),
        edge_attr=v,
        sparse_sizes=[s+rows_to_add for s in adj.sizes()]
    )

    newadj = add_sparse(newadj, combined_t)
    newadj = add_sparse(newadj, adj)

    return newadj

def threshold_sparse(s, threshold=0.5):
    '''
    Applies a threshold to each value in a sparse matrix. Returns a new SparseTensor
    filled with either 1.0s or 0.0s.
    Arguments:
    s (SparseTensor): 
    threshold (float):
    Returns:
    (SparseTensor)
    '''
    r, c, v = s.coo()
    v = torch.where(v>=threshold, 1.0, 0.0)
    return SparseTensor.from_edge_index(
        torch.stack((r,c),0), v, s.sparse_sizes()
    )

def assign_sparse_sub(adj, sub):
    '''
    Assign values of a tensor sub into a fragment of adj, overwriting adj with sub values
    Arguments:
    adj (SparseTensor): the main tensor (usually the larger)
    sub (SparseTensor): the tensor to be pasted onto adj (usually the smaller)
    Returns:
    (SparseTensor): with the shape of adj
    '''
    adj_r, adj_c, adj_v = adj.coo()
    sub_r, sub_c, sub_v = sub.coo()

    # remove from adj the elements that coincide with sub
    mask = [False if (r in torch.arange(3) and c in torch.arange(3)) else True for r, c in zip(adj_r, adj_c)]
    newadj_r = adj_r[mask]
    newadj_c = adj_c[mask]
    newadj_v = adj_v[mask]

    # append to adj the elements in sub
    newadj_r = torch.cat((newadj_r, sub_r), 0)
    newadj_c = torch.cat((newadj_c, sub_c), 0)
    newadj_v = torch.cat((newadj_v, sub_v), 0)

    return SparseTensor.from_edge_index(
        torch.stack([newadj_r, newadj_c], 0),
        newadj_v, 
        adj.sizes()
    )