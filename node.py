class Node:
    def __init__(self):
        self.depth = 0
        self.isContinuous = False
        self.isleaf = False
        self.feat = None
        #for continuous feature
        self.val = None
        self.category = None
        self.childlist = []
       
    def setleaf(self,category):
        self.isleaf = True
        self.childlist = []
        self.category = category
        return
    
    def add_child(self):
        self.childlist.append(Node())
        self.childlist[-1].depth = self.depth+1
        return

    