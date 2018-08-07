import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None
    
    def train(self, features: List[List[float]], labels: List[int]):
        # init.
        assert(len(features) > 0)
        self.feautre_dim = len(features[0])
        num_cls = np.max(labels)+1
        
        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()
        return
    
    def predict(self, features: List[List[float]]) -> List[int]:
        y_pred = []
        for feature in features:
            y_pred.append(self.root_node.predict(feature))
        return y_pred
    
    
    def print_tree(self, node=None, name='node 0', indent=''):
        if node is None:
            node = self.root_node
        print(name + '{')
        if node.splittable:
            print(indent + '  split by dim {:d}'.format(node.dim_split))
            for idx_child, child in enumerate(node.children):
                self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
        else:
            print(indent + '  cls', node.cls_max)
        print(indent+'}')


class TreeNode(object):
    def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label # majority of current node
        
        if len(np.unique(labels)) < 2 or len(np.unique(features)) < 2:
            self.splittable = False
        else:
            self.splittable = True
        
        self.dim_split = None # the dim of feature to be splitted
        self.feature_uniq_split = None # the feature to be splitted
    
    def split(self):
        temp = []
        for fea in self.features:
            temp.append(np.where(fea == [],0,1))
        if np.sum(temp) != 0:
            exit
        def conditional_entropy(branches: List[List[int]]) -> float:
            '''
            branches: C x B array, 
					 C is the number of classes,
					 B is the number of branches
					 it stores the number of 
            '''
        		 ########################################################
        		 # TODO: compute the conditional entropy
        		 ########################################################
            branches = np.array(branches)
            C,B = branches.shape
            store = np.zeros((C,B))
            colsum = np.sum(branches,axis=0)
            for i in range(C):
                for j in range(B):
                    store[i,j] = branches[i,j]/colsum[j]*np.log(branches[i,j]/colsum[j])
            store = np.nan_to_num(store)
            vec = -np.sum(store,axis = 0)     
            weight = colsum/np.sum(branches)
            H = np.inner(vec,weight)
            return H
        
        features = np.array(self.features)
        labels = np.array(self.labels)
        entropy_store = []
        C = self.num_cls
        unique_labels = np.arange(C)
        for idx_dim in range(len(self.features[0])):
            ############################################################
        		 # TODO: compare each split using conditional entropy
        		 #       find the best split
        		 ############################################################
            this_dim = features[:,idx_dim]
            unique_dim = np.unique(this_dim)
            B = len(unique_dim)
            to_branch = np.transpose(np.array([this_dim.tolist(),labels]))
            branch_temp = np.zeros((C,B))
            for p in range(C):
                for l in range(B):
                    branch_temp[p,l] = len(to_branch[(to_branch[:,0] == unique_dim[l]) & (to_branch[:,1] == unique_labels[p])])
            entropy_store.append(conditional_entropy(branch_temp))
            
            self.dim_split =  np.argmin(entropy_store)
            self.feature_uniq_split = np.unique(features[:,self.dim_split]).tolist()
        
		   ############################################################
		   # TODO: split the node, add child nodes
		   ############################################################
        # split the child nodes
            
        to_split = self.feature_uniq_split
        cut = self.dim_split
        for m in range(len(to_split)):
            subfeatures = features[features[:,cut] == to_split[m]]
            newfeatures = np.delete(subfeatures,cut,axis=1).tolist()
            newlabels = labels[features[:,cut] == to_split[m]].tolist()
            
            chil = TreeNode(newfeatures,newlabels,self.num_cls)
            self.children.append(chil)
               
        for child in self.children:
            if child.splittable:
                child.split()                
        return

    def predict(self, feature: List[int]) -> int:
        if self.splittable:
            idx_child = self.feature_uniq_split.index(feature[self.dim_split])
            feature = feature[:self.dim_split] + feature[self.dim_split + 1:]
            return self.children[idx_child].predict(feature)
        else:
            return self.cls_max



