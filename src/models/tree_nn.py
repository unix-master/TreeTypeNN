
import numpy as np
# from scipy.optimize import linprog
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from models.shared_trunk import SharedTrunk

print_lock = threading.Lock()

class TreeNode:
    def __init__(self,in_dim,hidden_dim,childA:TreeNode=None,childB:TreeNode=None):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        ##? parameters for a 2-layer neural network
        self.W1 = np.random.randn(in_dim,hidden_dim) * np.sqrt(1/in_dim) #kaiming
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim,1) * np.sqrt(1/hidden_dim)
        self.b2 = np.zeros(1)
        ##
        self.childA = childA
        self.childB = childB
        ##??
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.t = 0

        self.mW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2)

        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)

    def _forward(self,X):
        def relu(x):
            return np.maximum(0.0,x)
        self.z1 = X@self.W1 + self.b1
        self.h = relu(self.z1)
        self.z2 = self.h@self.W2 + self.b2
        return self.z2.ravel()

    def _backward(self,p,X,y):
        #?? BCE loss 2-layer neural network
        dz2 = (p-y).reshape(-1,1)
        dz2 /= len(X)
        self.dW2 = self.h.T@dz2
        self.db2 = np.sum(dz2,axis=0)
        dh = dz2@self.W2.T
        dz1 = dh * (self.z1 > 0)
        self.dW1 = X.T @ dz1
        self.db1 = np.sum(dz1,axis=0)

    def step(self,lr):
        # self.W1 -= lr*self.dW1
        # self.W2 -= lr*self.dW2
        # self.b1 -= lr*self.db1
        # self.b2 -= lr*self.db2
        self.t += 1
        params = ("W1", "b1", "W2", "b2")
        for name in params:
            param = getattr(self, name)
            grad = getattr(self, "d" + name)

            m_name = "m" + name
            m = getattr(self, m_name)
            m = self.beta1 * m + (1 - self.beta1) * grad
            setattr(self, m_name, m)

            v_name = "v" + name
            v = getattr(self, v_name)
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
            setattr(self, v_name, v)

            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)

            param -= lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def calculate_activation(self,X):
        features = X
        if self.childA is not None:
            net = self.childA.calculate_activation(X)
            xA = (net>=0).astype(float)
            features = np.column_stack([features,xA])
        if self.childB is not None:
            net = self.childB.calculate_activation(X)
            xB = (net>=0).astype(float)
            features = np.column_stack([features,xB])

        z2 = self._forward(features)
        return z2

    def num_parameters(self):
        return (self.W1.size+self.b1.size+self.W2.size+self.b2.size)       

class MultiClassTreeNN:
    def __init__(self,num_classes,hidden_dim,shared_hidden_dim,shared_feature_dim,dataset:str,shared_epochs=10,lr=0.01,batch_size=64,epochs=100,max_depth=5):
        self.num_classes = num_classes
        self.max_depth = max_depth
        self.hidden_dim = hidden_dim
        self.shared_hidden_dim = shared_hidden_dim
        self.shared_feature_dim = shared_feature_dim
        self.parents = [None]*num_classes
        self.lr = lr; self.epochs = epochs; self.batch_size = batch_size
        self.shared_epochs = shared_epochs
        self.dataset = dataset

    #/ get total number of parameters in the tree
    def num_parameters(self):
        def _parameters(node:TreeNode):
            if node is None:
                return 0
            total = node.num_parameters()
            total += _parameters(node.childA)
            total += _parameters(node.childB)
            return total
        return sum(_parameters(p) for p in self.parents)

    def _train_node(self,X,y) -> TreeNode:
        N, n = X.shape
        root = TreeNode(in_dim=n,hidden_dim=self.hidden_dim)
        # training iterations
        for _ in range(self.epochs):
            # if self.batch_size >= N:
            #     indices = np.arange(N)
            # else:
            #     indices = np.random.choice(N,size=self.batch_size,replace=False)
            for i in range(0,N,self.batch_size):
                j = min(i+self.batch_size,N)
                X_batch = X[i:j]; y_batch = y[i:j]
                z2 = root._forward(X_batch)
                z2 = np.clip(z2,-50,50)
                p = 1.0 / (1.0 + np.exp(-z2))
                ##! compute gradients for a 2-layer network
                root._backward(p,X_batch,y_batch)
                root.step(lr=self.lr)
                # print(f'Train accuracy: {(np.mean(y_pred == y)):.4f}')
        return root

    def _build_tree(self,X,y,k,depth=0,max_depth=5):
        with print_lock:
            print(f'Tree {k}: Current Depth = {depth}',end='\r',flush=True)
        N,n = X.shape
        node = self._train_node(X,y)
        # if w is not None:
        #     return TreeNode(w=w,b=b)
        if np.all(y==y[0]) or depth == max_depth:
            return node
        ##??
        y_shifted = (2*y-1).flatten()
        margin = y_shifted*node._forward(X)
        fnmask = (y_shifted==1)&(margin<0)
        tnmask = (y_shifted==-1)&(margin>=0)
        fpmask = (y_shifted==-1)&(margin<0)
        tpmask = (y_shifted==1)&(margin>=0)
        childA = None
        # xA = np.zeros(N)
        if len(X[fnmask])>0:
            # assert False, f"{X[fnmask].shape}, {X[tnmask].shape}"
            Xa = np.vstack((X[fnmask],X[tnmask]))
            ya = np.hstack((np.ones(len(X[fnmask])),np.zeros(len(X[tnmask]))))
            if len(Xa)<N and depth < self.max_depth:
                childA = self._build_tree(Xa,ya,k,depth+1,max_depth)
        childB = None
        # xB = np.zeros(N)
        if len(X[fpmask])>0:
            Xb = np.vstack((X[fpmask],X[tpmask]))
            yb = np.hstack((np.ones(len(X[fpmask])),np.zeros(len(X[tpmask]))))
            if len(Xb)<N or depth < self.max_depth:
                childB = self._build_tree(Xb,yb,k,depth+1,max_depth)
        node.childA = childA; node.childB = childB
        child_features = []
        if childA is not None:
            netA = childA.calculate_activation(X)
            xA = (netA>=0).astype(float)
            child_features.append(xA.reshape(-1,1))
        if childB is not None:
            netB = childB.calculate_activation(X)
            xB = (netB>=0).astype(float)
            child_features.append(xB.reshape(-1,1))
        if len(child_features)==0:
            return node
        X_combined = np.column_stack([X]+child_features)
        node = self._train_node(X_combined,y)
        node.childA = childA
        node.childB = childB
        return node

    def fit(self,X,y):
        N, n = X.shape
        self.trunk = SharedTrunk(dataset=self.dataset,in_dim=n,hidden_dim=self.shared_hidden_dim,out_dim=self.shared_feature_dim,num_classes=self.num_classes)
        if not self.trunk.load(self.shared_epochs): #? load saved weights if exists
            self.trunk.fit(X,y,epochs=self.shared_epochs,lr=self.lr,batch_size=self.batch_size)
            self.trunk.save()
        else:
            print('Loaded shared trunk saved weights!')
        X = self.trunk.map_features(X,batch_size=self.batch_size)
        # assert False, X.shape
        print('Training tree neural network with shared features...')
        def _solve(X,y,k):
            y_k = (y==k).astype(float) #! one-vs-rest setup
            return k, self._build_tree(X,y_k,k,depth=0,max_depth=self.max_depth)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_solve,X,y,k) for k in range(self.num_classes)]
            for future in as_completed(futures):
                k, tree = future.result()
                self.parents[k] = tree
        # for k in range(self.num_classes):
        #     _, self.parents[k] = _solve(X,y,k)

    def predict_scores(self,X):
        N,n = X.shape
        scores = np.zeros((N,self.num_classes))
        X = self.trunk.map_features(X,batch_size=self.batch_size)

        def _eval(k):
            # print(f'Start predict:Tree {k}')
            node: TreeNode = self.parents[k]
            # print(f'Node {k} created')
            net = node.calculate_activation(X)
            # print(f'Finish predict:Tree {k}')
            return k,net

        with ThreadPoolExecutor(max_workers=min(self.num_classes,4)) as executor:
            futures = [executor.submit(_eval,k) for k in range(self.num_classes)]
            for future in as_completed(futures):
                k, net = future.result()
                scores[:,k] = net
        # for k in range(self.num_classes):
        #     _, scores[:,k] = _eval(k)
        return scores

    def loss(self,X,y):
        scores = self.predict_scores(X)
        #?/ for binary classification (i'll just do it for now)
        scores = scores[:,1]
        loss = np.mean(np.mean(np.maximum(0.0,scores)-scores*y+np.log1p(np.exp(-np.abs(scores)))))
        return loss

    def predict(self,X):
        scores = self.predict_scores(X)
        if self.num_classes==2:
            return (scores[:,1] >= 0).astype(float)
        #: scores = logits
        return np.argmax(scores,axis=1)
        