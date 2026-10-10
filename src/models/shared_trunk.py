import numpy as np
import tqdm
import math

class SharedTrunk:
    #? for shared feature mapping initially
    def __init__(self,in_dim,hidden_dim,out_dim,num_classes,dataset:str):
        self.in_dim = in_dim
        self.out_dim = out_dim # will use it to map to the feature dimension
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.W1 = np.random.randn(in_dim,hidden_dim)*np.sqrt(1/in_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim,out_dim)*np.sqrt(1/hidden_dim)
        self.b2 = np.zeros(out_dim)
        #! classifier head : since i'm not training this end-to-end with tree
        self.Wout = np.random.randn(out_dim,num_classes)*np.sqrt(1/out_dim)
        self.bout = np.zeros(num_classes)
        self.dataset = dataset

    def _forward(self,X):
        def relu(x):
            return np.maximum(0.0,x)
        z1 = X@self.W1 + self.b1
        h1 = relu(z1)
        z2 = h1@self.W2 + self.b2
        h2 = relu(z2)
        out = h2@self.Wout + self.bout
        # assert False, out.shape
        return z1,h1,z2,h2,out

    def _backward(self,p,X,y,z1,h1,z2,h2):
        #? dout/
        # N,_ = X.shape
        dout = (p - y)/X.shape[0]
        self.dWout = h2.T@dout
        self.dbout = np.sum(dout,axis=0)
        dh2 = dout@self.Wout.T
        dz2 = dh2 * (z2>0) # because of relu
        self.dW2 = h1.T@dz2
        self.db2 = np.sum(dz2,axis=0)
        dh1 = dz2@self.W2.T
        dz1 = dh1*(z1>0)
        self.dW1 = X.T@dz1
        self.db1 = np.sum(dz1,axis=0)

    def step(self,lr):
        # self.W1 -= lr*self.dW1
        # self.b1 -= lr*self.db1
        # self.W2 -= lr*self.dW2
        # self.b2 -= lr*self.db2
        # self.Wout -= lr*self.dWout
        # self.bout -= lr*self.dbout
        #?? ended up switching to adam from standard gradient descent
        beta1 = 0.9; beta2 = 0.999; eps = 1e-8
        params = {
            "W1": self.W1, "b1": self.b1,
            "W2": self.W2, "b2": self.b2,
            "Wout": self.Wout, "bout": self.bout
        }
        grads = {
            "W1": self.dW1, "b1": self.db1,
            "W2": self.dW2, "b2": self.db2,
            "Wout": self.dWout, "bout": self.dbout
        }
        if not self.adam_m:
            self.adam_m = {k: np.zeros_like(v) for k, v in params.items()}
            self.adam_v = {k: np.zeros_like(v) for k, v in params.items()}
        self.adam_t += 1
        t = self.adam_t
        for k, param in params.items():
            g = grads[k]
            self.adam_m[k] = beta1 * self.adam_m[k] + (1 - beta1) * g
            self.adam_v[k] = beta2 * self.adam_v[k] + (1 - beta2) * g**2
            m_hat = self.adam_m[k] / (1 - beta1**t)
            v_hat = self.adam_v[k] / (1 - beta2**t)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def fit(self,X,y,batch_size,epochs=100,lr=0.01):
        #? for adam optimizer
        self.adam_t = 0
        self.adam_m = {}; self.adam_v = {}
        #
        self.train_epochs = epochs
        ytilde = y.copy()
        y = np.eye(self.num_classes,dtype=np.float64)[y.astype(int)]
        print(f'Training shared trunk...')
        N,n = X.shape #/ assuming flattened feature space
        for epoch in range(epochs):
            perm = np.random.permutation(N)
            print(f'Epoch {epoch+1} ',end='')
            n_batches = math.ceil(len(perm)/batch_size)
            loader = tqdm.tqdm(range(0,N,batch_size),desc=f'Epoch {epoch+1} -- Batch [0/{n_batches}]',bar_format='{desc} {bar} {postfix}',leave=False)
            for idx, i in enumerate(loader):
                indices = perm[i:i+batch_size]
                X_batch = X[indices]; y_batch = y[indices]
                z1,h1,z2,h2,out = self._forward(X_batch)
                #! softmax
                # shift = out - np.max(out,axis=1,keepdims=True)
                # exps = np.exp(shift)
                # p = exps/np.sum(exps,axis=1,keepdims=True)
                out = np.clip(out,-50,50)
                p = 1/(1+np.exp(-out))
                #
                self._backward(p,X_batch,y_batch,z1,h1,z2,h2)
                self.step(lr)
                #
                # if epoch == 0 or epoch+1 % 10==0:
                loader.set_description(f'Epoch {epoch+1} Batch [{idx+1}/{n_batches}]')
                # loader.set_postfix(accuracy=f'{acc:.4f}')
        out = self.predict_score(X)
        y_pred = np.argmax(out,axis=1)
        acc = np.mean(ytilde==y_pred)
        print(f'Train accuracy: {acc:.4f}')

    def predict_score(self,X):
        def relu(x):
            return np.maximum(0.0,x)
        z1 = X@self.W1 + self.b1
        h1 = relu(z1)
        z2 = h1@self.W2 + self.b2
        h2 = relu(z2)
        return h2@self.Wout+self.bout

    def map_features(self,X,batch_size):
        N,n = X.shape
        def relu(x):
            return np.maximum(0.0,x)
        features = np.empty((N,self.out_dim))
        for i in range(0,N,batch_size):
            j = min(i+batch_size,N)
            X_batch = X[i:j]
            z1 = X_batch@self.W1+self.b1
            h1 = relu(z1)
            z2 = h1@self.W2+self.b2
            h2 = relu(z2)
            features[i:j] = h2
        return features

    def save(self):
        import os
        os.makedirs('src/saved',exist_ok=True)
        np.savez(f'src/saved/{self.dataset}_{self.hidden_dim}_{self.out_dim}_{self.train_epochs}.npz',self.W1,self.b1,self.W2,self.b2,self.Wout,self.bout)

    def load(self,epochs) -> bool:
        import os
        if os.path.isfile(f'src/saved/{self.dataset}_{self.hidden_dim}_{self.out_dim}_{epochs}.npz'):
            data = np.load(f'src/saved/{self.dataset}_{self.hidden_dim}_{self.out_dim}_{epochs}.npz',mmap_mode='r')
            # assert False, f'{self.W1.shape}'
            self.W1 = data['arr_0']
            self.b1 = data['arr_1']
            self.W2 = data['arr_2']
            self.b2 = data['arr_3']
            self.Wout = data['arr_4']
            self.bout = data['arr_5']
            # assert False, data
            #?? why do i have to do it like this??
            return True
        return False