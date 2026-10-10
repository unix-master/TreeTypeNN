## tree-type neural network for images
import sys
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

import os
import cv2
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from models.tree_nn import MultiClassTreeNN

def train_test_split(X: np.ndarray,y: np.ndarray,split=0.8,seed=42):
    rng = np.random.default_rng(seed)
    n_samples = X.shape[0]
    n_train = int(n_samples*split)
    indices = rng.choice(n_samples,size=n_train,replace=False)
    X_train = X[indices]; y_train = y[indices]
    X_test = X[~indices]; y_test = y[~indices]
    return X_train, y_train, X_test, y_test

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=float, default=1.0)
    parser.add_argument('--dataset',type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = _parse_args()
    dataset = args.dataset
    if dataset is None:
        assert False, 'Dataset is not provided.'
    if dataset == 'catsvsdogs':
        datadir = 'data/catsvsdogs'
    elif dataset == 'cifar10':
        datadir = 'data/cifar10/cifar-10-batches-py'

    if dataset == 'catsvsdogs':
        X, y = [], []
        classes = [f for f in os.listdir(datadir) if not f.startswith('.')]
        for c in classes:
            dir = os.path.join(datadir,c)
            class_idx = classes.index(c)
            for img_path in os.listdir(dir):
                # got all the images here
                img = cv2.imread(os.path.join(dir,img_path))
                if img is not None:
                    img = cv2.resize(img,(128,128))
                    X.append(img)
                    y.append(class_idx)
        X = np.array(X); y = np.array(y)
        # print(f'{X.shape}, {y.shape}')
        X_train, y_train, X_test, y_test = train_test_split(X,y,split=0.8,seed=2312)
    elif dataset == 'cifar10':
        classes = range(10)
        def load_data(filepath):
            # assert False, filepath
            with open(filepath,'rb') as f:
                data_dict = pickle.load(f,encoding='bytes')
            X = data_dict[b'data']
            X = X.reshape((X.shape[0], 3, 32, 32))
            X = X.transpose(0, 2, 3, 1)
            y = np.array(data_dict[b'labels'])
            return X,y

        xs, ys = [], []
        for i in range(1, 6):
            filepath = os.path.join(datadir, f"data_batch_{i}")
            X,y = load_data(filepath)
            xs.append(X)
            ys.append(y)

        X_train = np.concatenate(xs, axis=0)
        y_train = np.concatenate(ys, axis=0)

        test_path = os.path.join(datadir, "test_batch")
        X_test, y_test = load_data(test_path)

    X_train = X_train / 255.0; X_test = X_test / 255.0 
    # assert False, classes

    # if args.limit < 1.0:
    #     assert type(args.limit) is float
    #     limit_size = int(X.shape[0]*args.limit)
    #     # assert False, limit_size
    #     X = X[:limit_size]
    #     y = y[:limit_size]

    # print(f'train: {X_train.shape}, test: {X_test.shape}')
    ##?? for now, i'll just flatten the features
    X_train_flatten = X_train.reshape(X_train.shape[0],-1)
    X_test_flatten = X_test.reshape(X_test.shape[0],-1)

    #! let's experiment with node hidden dimension
    train_loss, test_loss = [], []
    parameters = []
    node_hidden_dims = [2,8,16,32,64,128,256,512]

    for d in node_hidden_dims:
        model = MultiClassTreeNN(
            dataset=args.dataset,
            num_classes=len(classes),
            hidden_dim=d,
            max_depth=3,
            shared_hidden_dim=256,
            shared_feature_dim=128,
            shared_epochs=100,
            epochs=150,
            lr=0.001,
            batch_size=128
        )

        model.fit(X_train_flatten,y_train)
        y_train_pred = model.predict(X_train_flatten)
        y_train_loss = model.loss(X_train_flatten,y_train)
        acc_train = np.mean(y_train == y_train_pred)
        y_test_pred = model.predict(X_test_flatten)
        y_test_loss = model.loss(X_test_flatten,y_test)
        acc_test = np.mean(y_test == y_test_pred)
        #??
        train_loss.append(y_train_loss); test_loss.append(y_test_loss)
        n_parameters = model.num_parameters()
        parameters.append(f'{(n_parameters/1000):.1f}K')
        print(f'----- Node hidden dimension = {d} -----')
        print(f'Train accuracy: {acc_train:.4f}')
        print(f'Test accuracy: {acc_test:.4f}')

    #double descent plot
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,8))
    ax.plot(n_parameters,train_loss,label='Train loss')
    ax.plot(n_parameters,test_loss,label='Test loss')
    ax.set_xlabel('Tree network size (#parameters)')
    ax.set_xticks(n_parameters)
    ax.set_xticklabels(parameters)
    ax.grid(alpha=0.5)
    fig.savefig('double_descent_catsvsdogs.png',bbox_inches='tight')
    plt.close(fig)
