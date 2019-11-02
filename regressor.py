import numpy as np
import pandas as pd
from functools import partial
class Node:
    def __init__(self,depth = 0):
        self.depth = depth
        self.leftchild = None
        self.rightchild = None
        self.isleaf = False
        self.feat = None
        #for continuous feature
        self.val = None
        self.mean = None
        
class Regressor:
    def __init__(self,max_depth = None):
        self.max_depth = max_depth
        self.root = Node()
    def fit(self,x:np.ndarray,y:np.ndarray):
        self.generate(self.root,x,y)
        
    def generate(self,node,x,y):
        
        node.mean = y.mean()
        if(self.max_depth!=None):
            if(node.depth>=self.max_depth-1):
                node.isleaf = True
        if(y.shape[0]==1):
            node.isleaf = True
            return

        best_val_dict = {}
        #feat -> best splitting val of that feat
        for feat in range(x.shape[0]):
            judging_vals = partial(self.loss,x,y,feat)
            #find the best splitting point for one feature
            best_val_dict[feat] = min(x[feat],key = judging_vals)
        
        
        best_feat = min(range(x.shape[0]),key = lambda feat:self.loss(x,y,feat = feat,val = best_val_dict[feat]))
        best_val = best_val_dict[best_feat]
        
        node.feat = best_feat
        node.val = best_val
        
        
        y_left = y[x[node.feat]<=node.val]
        y_right = y[x[node.feat]>node.val]
    
        node.leftchild = Node(node.depth+1)
        self.generate(node.leftchild,x[:,x[best_feat]<=best_val],y_left)
        
        node.rightchild = Node(node.depth+1)
        self.generate(node.rightchild,x[:,x[best_feat]>best_val],y_right)

        
        return
    def squared_error(self,y):
        if(y.shape[0]==0):
            return 0
        else:
            return y.var()*y.shape[0]
    
    
    def loss(self,x,y,feat,val):
        return self.squared_error(y[x[feat]<=val])+self.squared_error(y[x[feat]>val])
    
    def pred(self,x):
        return self.traversal(x,self.root)
    
    def traversal(self,x,node):
        if(node.isleaf):
            return node.mean
        else:
            if(x[node.feat]<=node.val):
                 return   self.traversal(x,node.leftchild)
            else:
                return self.traversal(x,node.rightchild)
    def plotting(self,node):
        '''
        Getting name of feature from parent node,geting responding value from child node      
        '''
        if(node.isleaf):
            print('|   '*node.depth,end = '')
            print("|--- out:{}".format(node.mean))
            return
        else:
            print('|   '*node.depth,end = '')
            print("|--- x{} <= {}".format(node.feat,node.val))
            self.plotting(node.leftchild)
            print('|   '*node.depth,end = '')
            print("|--- x{} > {}".format(node.feat,node.val))
            self.plotting(node.rightchild)
            
        return
        