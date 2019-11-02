import numpy as np
import pandas as pd
from copy import deepcopy
from functools import partial

from node import*
from utils import*


class Classifier:
    def __init__(self,criterion,pruning = None):
        assert criterion in ["gini","entropy","ratio"]
        if(pruning!=None):
            assert pruning in ["post","pre"]
        self.criterion = criterion
        self.isContinuous = {}
        self.feat_list = []
        self.feat_vals = {}
        self.best_points = {}
        self.root = Node()
        #change
        self.splitting_points = {}
        
        self.train_x = None
        self.train_y = None
        self.dev_x = None
        self.dev_y = None
        self.pruning = pruning
        
        #self.node_cache = []

    def fit(self,x,y,dev_x = None,dev_y = None,split = False,split_ratio = 0.7,shuffle = False):
        #x:pandas.Dataframe,y:pandas.Series
        
       
        
        if(self.pruning or split):
            if(type(dev_x) == type(None) or type(dev_y) == type(None)):
                #If pruning is required but no dev set is given,
                #We'll split input samples into train set and dev set
                train_idx,dev_idx = train_dev_spiltting(x.shape[0],ratio = split_ratio,shuffle = shuffle)
                self.train_x = x.iloc[train_idx]
                self.train_y = y.iloc[train_idx]
                self.dev_x = x.iloc[dev_idx]
                self.dev_y = y.iloc[dev_idx]
            else:
                self.train_x = x
                self.train_y = y
                self.dev_x = dev_x
                self.dev_y = dev_y
        else:
            self.train_x = x
            self.train_y = y
        self.feat_list = list(self.train_x.columns)
        for feat in self.feat_list:
            if(self.train_x[feat].dtype == 'float32' or self.train_x[feat].dtype == 'float64'):
                self.isContinuous[feat] = True
                #change
                self.splitting_points[feat] = self.get_splitting_points(x[feat])
            else:
                self.isContinuous[feat] = False
                if(self.pruning == None or type(dev_x) == type(None) or type(dev_y) == type(None)):
                    self.feat_vals[feat] = list(set(x[feat]))
                else:       
                    self.feat_vals[feat] = list(set(self.train_x[feat]).union(set(self.dev_x[feat])))
        '''
        if(self.pruning != 'pre'):
            self.generate(self.root,self.train_x,self.train_y,self.feat_list)
        elif(self.pruning == 'pre'):
            self.generate(self.root,self.train_x,self.train_y,self.feat_list)
        '''
        self.generate(self.root,self.train_x,self.train_y,self.feat_list)
        if(self.pruning == 'post'):
            self.postpruning()
        
    def all_the_same(self,x):
        #to be optimized
        return x[x==x.iloc[0]].shape[0] == x.shape[0]
    
    
    def generate(self,node:Node,x,y,feats):
        #Recurrently generate the tree from root node.
        #Assuming x is not empty
        
        #print(len(node.childlist))

        #All samples in x belongs to the same class
        
        
        
        node_cache = []
        y_vals = set(y)
        if(len(y_vals)==1):
            node.setleaf(y_vals.pop())
            return
        
        
        most_category = max(y_vals,key = lambda y_val:y[y==y_val].shape[0])
        
        #If the feature set is empty,
        #or all samples in x have the same value over every feature. 
        if(not feats):
            node.setleaf(most_category)
            return
        
        ret = True
        for feat in feats:
            ret = ret and self.all_the_same(x[feat])
            if(not ret):
                break
        if(ret):
            node.setleaf(most_category)
            return
        #preparing for pruning
        node.category = most_category
        best_feat = self.get_best_features(x,y,feats)
        node.feat = best_feat
        before_acc = 0
        
       
        if(self.pruning == 'pre'):
            node.setleaf(most_category)
            before_acc = self.eval(self.dev_x,self.dev_y)
            node.isleaf = False

        
        #Continuous feature
        if(self.isContinuous[best_feat]):
            node.val = self.best_points[best_feat]
            #<=
            node.add_child()
            if(x[x[best_feat]<=node.val].shape[0] == 0):
                node.childlist[-1].setleaf(node.category)
            else:
                if(self.pruning == 'pre'):
                    node.childlist[-1].setleaf(max(set(y[x[best_feat]<=node.val]),
                                     key = lambda y_val:y[y==y_val].shape[0]))
                    node_cache.append([0,node.childlist[-1]])
                else:    
                    self.generate(node.childlist[-1],
                                            x[x[best_feat]<=node.val],y[x[best_feat]<=node.val],feats)
            #>
            node.add_child()
            if(x[x[best_feat]>node.val].shape[0] == 0):
                node.childlist[-1].setleaf(node.category)
            else:
                if(self.pruning == 'pre'):
                    node.childlist[-1].setleaf(max(set(y[x[best_feat]>node.val]),
                                     key = lambda y_val:y[y==y_val].shape[0]))
                    node_cache.append([1,node.childlist[-1]])
                else:    
                    self.generate(node.childlist[-1],
                                  x[x[best_feat]>node.val],y[x[best_feat]>node.val],feats)
        #Discrete feature
        else:
            for idx ,feat_val in enumerate(self.feat_vals[best_feat]):
                node.add_child()
                if(x[x[best_feat]==feat_val].shape[0] == 0):
                    node.childlist[-1].setleaf(node.category)
                else:
                    if(self.pruning == 'pre'):
                        node.childlist[-1].setleaf(max(set(y[x[best_feat] == feat_val]),
                                     key = lambda y_val:y[y==y_val].shape[0]))
                        node_cache.append([idx,node.childlist[-1]])
                    else:
                        rest_feats = list(feats)
                        rest_feats.remove(best_feat)
                        self.generate(node.childlist[-1],x[x[best_feat]==feat_val],
                                         y[x[best_feat]==feat_val],rest_feats)
                        
                        
        #evaluating after  generating  subtree
        if(self.pruning == 'pre'):
            after_acc = self.eval(self.dev_x,self.dev_y)
            #print(before_acc,after_acc)
            if(before_acc>after_acc):
                #stop generating
                node.setleaf(most_category)
                return 
            else:
                #keep generating
                if(self.isContinuous[best_feat]):
                    for idx_node in node_cache:
                        idx = idx_node[0]
                        child = idx_node[1]
                        if(idx == 0):
                            self.generate(child,x[x[best_feat]<=node.val],y[x[best_feat]<=node.val],feats)
                        else:
                            self.generate(child,x[x[best_feat]>node.val],y[x[best_feat]>node.val],feats)
                else:
                    rest_feats = list(feats)
                    rest_feats.remove(best_feat)
                    for idx_node in node_cache:
                        idx = idx_node[0]
                        child = idx_node[1]
                        self.generate(child,x[x[best_feat]==(self.feat_vals[best_feat])[idx]],
                                               y[x[best_feat]==(self.feat_vals[best_feat])[idx]],
                                               rest_feats)
                            
                        
        return
        
    
    def get_best_features(self,x,y,feats):
        if(self.criterion == "gini"):
            judge = partial(self.gini_index,x,y)
            return min(feats,key = judge)
        elif(self.criterion == "entropy"):
            judge = partial(self.info_gain,x,y)
            return max(feats,key = judge)
        elif(self.criterion == "ratio"):
            judge = partial(self.gain_ratio,x,y)
            return max(feats,key = judge)
    def get_splitting_points(self,x):
        #x[feat]
        ordered = list(set(x))
        ordered.sort()
        points = []
        if(len(ordered)==1):
            points.append(ordered[0])
            return points
        for a,b in pairwise(ordered):
            points.append(a+(b-a)/2)
        return points
    
    def gain_ratio(self,x,y,feat):
        #So far it works for discrete features only
        return self.info_gain(x,y,feat)/self.entropy(x[feat])
        
    
    def gini(self,y):
        total = 1
        y_vals = set(y)
        for y_val in y_vals:
            pk = y[y==y_val].shape[0]/y.shape[0]
            total -= pk**2
        return total
    
    def gini_index(self,x,y,feat):
        gini_index = 0
        if(not self.isContinuous[feat]):
            feat_vals = set(x[feat])
            for feat_val in feat_vals:
                gini_index += x[x[feat]==feat_val].shape[0]/x.shape[0]*self.gini(y[x[feat]==feat_val])
            return gini_index
        else:
            #change
            splitting_points = self.splitting_points[feat]
            #gini_index = max()
            best_point = min(splitting_points,key = lambda point:\
                             + x[x[feat]<=point].shape[0]/x.shape[0]*self.gini(y[x[feat]<=point])\
                             + x[x[feat]>point].shape[0]/x.shape[0]*self.gini(y[x[feat]>point]))
            gini_index =  \
            + x[x[feat]<=best_point].shape[0]/x.shape[0]*self.gini(y[x[feat]<=best_point])\
            + x[x[feat]>best_point].shape[0]/x.shape[0]*self.gini(y[x[feat]>best_point])
            #to be optimized
            self.best_points[feat] = best_point
            return gini_index 
                

    def entropy(self,y):
        total = 0
        y_vals = set(y)
        for y_val in y_vals:
            pk = y[y==y_val].shape[0]/y.shape[0]
            total -= pk*np.log2(pk)
        return total
    
    def info_gain(self,x,y,feat):
        info_gain = self.entropy(y)
        if(not self.isContinuous[feat]):
            feat_vals = set(x[feat])
            for feat_val in feat_vals:
                info_gain -= x[x[feat]==feat_val].shape[0]/x.shape[0]*self.entropy(y[x[feat]==feat_val])
            return info_gain
        else:
            #splitting_points = self.get_splitting_points(x[feat])
            splitting_points = self.splitting_points[feat]
            best_point = max(splitting_points,key = lambda point:info_gain \
                             - x[x[feat]<=point].shape[0]/x.shape[0]*self.entropy(y[x[feat]<=point])\
                             - x[x[feat]>point].shape[0]/x.shape[0]*self.entropy(y[x[feat]>point]))
            info_gain = info_gain \
            - x[x[feat]<=best_point].shape[0]/x.shape[0]*self.entropy(y[x[feat]<=best_point])\
            - x[x[feat]>best_point].shape[0]/x.shape[0]*self.entropy(y[x[feat]>best_point])
            self.best_points[feat] = best_point 
            return info_gain
            #to be optimized
    def predict(self,X):
        #predicting the class for the whole set
        func = partial(self.pred,self.root)
        y_pred = X.apply(func,axis = 1)
        return y_pred
    def pred(self,node,x):
        #predicting the class of one single sample
        if(node.isleaf):
            return node.category
        else:
            if(self.isContinuous[node.feat]):
                if(x[node.feat]<=node.val):
                    return self.pred(node.childlist[0],x)
                else:
                    return self.pred(node.childlist[1],x)
            else:
                return self.pred(node.childlist[self.feat_vals[node.feat].index(x[node.feat])],x)
        
    def eval(self,x,y):
        #predict and evalutate
        count = 0 
        y_pred = self.predict(x)
        for i in range(y.shape[0]):
            if(y.iloc[i]==y_pred.iloc[i]):
                count+=1
        acc = count/y.shape[0]
        return acc
    
    def traversal(self,node,tree_stack):
        if(node.isleaf):
            #There is no need to replace node leaf
            return
        else:
            tree_stack.append(node)
            for child in node.childlist:
                self.traversal(child,tree_stack)

        
    def postpruning(self):
        tree_stack = []
        self.traversal(self.root,tree_stack)
        tree_stack.reverse()
        for node in tree_stack:
            self.attempting(node)
        return 

            
    def attempting(self,node):
        temp = deepcopy(node)
        before_acc = self.eval(self.dev_x,self.dev_y)
        #self.plotting(self.root)
        node.setleaf(node.category)
        after_acc = self.eval(self.dev_x,self.dev_y)
        #print(before_acc,after_acc)
        if(after_acc>=before_acc):
            #It's better to prune this subtree
            return
        else:
            #recover to previous state
            node.__dict__.update(temp.__dict__)
            #What the heck
            return
        
    
    def plotting(self,node):
        '''
        Getting name of feature from parent node,geting responding value from child node      
        '''
        if(node.isleaf):
            print('|   '*node.depth,end = '')
            print("|--- class:{}".format(node.category))
            return
        else:
            if(self.isContinuous[node.feat]):
                print('|   '*node.depth,end = '')
                print("|--- {} <= {}".format(node.feat,node.val))
                self.plotting(node.childlist[0])
                print('|   '*node.depth,end = '')
                print("|--- {} > {}".format(node.feat,node.val))
                self.plotting(node.childlist[1])
            else:
                for idx,child in enumerate(node.childlist):
                    print('|   '*node.depth,end = '')
                    print("|--- {} = {}".format(node.feat,self.feat_vals[node.feat][idx]))
                    self.plotting(child)   
        return
                
            
        