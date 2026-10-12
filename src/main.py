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
from utils import compute_occlusion_map, plot_node_occlusion

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
    num_parameters, parameters = [], []
    node_hidden_dims = [2,8,16,32,64,128,256,512]
    tree_depths = range(1,9)

    model = MultiClassTreeNN(
        dataset=args.dataset,
        num_classes=len(classes),
        hidden_dim=32,
        max_depth=4,
        shared_hidden_dim=256,
        shared_feature_dim=128,
        shared_epochs=100,
        epochs=150,
        lr=0.001,
        batch_size=128
    )
    model.fit(X_train_flatten,y_train)

    #!! check node sensitivity for rootk and its children
    # image_idx = 0
    # img_test = X_test[image_idx]
    # img_class = y_test[image_idx]
    # rootk = model.parents[img_class]
    #? calculate score
    # def score_fn(node):
    #     def _score(img_flat):
    #         img_features = model.trunk.map_features(img_flat,batch_size=model.batch_size)
    #         scores = node.calculate_activation(img_features)
    #         return np.asarray(scores).reshape(-1)
    #     return _score
    #?
    #! check sensitivity for rootk
    # score1, sensitivity1 = compute_occlusion_map(img_test,score_fn(rootk))
    # title1 = f'tree {img_class}, root, logit={score1:.3f}'
    # fig1 = plot_node_occlusion(img_test,sensitivity1,title=title1)
    # fig1.savefig(f'{dataset}_occlusion_root_class{img_class}.png',bbox_inches='tight')
    # plt.close(fig1)
    # if rootk.childA is not None:
    #     score2, sensitivity2 = compute_occlusion_map(img_test,score_fn(rootk.childA))
    #     title2 = f'tree {img_class}, root.childA, logit={score2:.3f}'
    #     fig2 = plot_node_occlusion(img_test,sensitivity2,title=title2)
    #     fig2.savefig(f'{dataset}_occlusion_child1_class{img_class}.png',bbox_inches='tight')
    #     plt.close(fig2)
    # if rootk.childB is not None:
    #     score3, sensitivity3 = compute_occlusion_map(img_test,score_fn(rootk.childB))
    #     title3 = f'tree {img_class}, root.childB, logit={score3:.3f}'
    #     fig3 = plot_node_occlusion(img_test,sensitivity3,title=title3)
    #     fig3.savefig(f'{dataset}_occlusion_child2_class{img_class}.png',bbox_inches='tight')
    #     plt.close(fig3)

    for d in tree_depths:
        model = MultiClassTreeNN(
            dataset=args.dataset,
            num_classes=len(classes),
            hidden_dim=64,
            max_depth=d,
            shared_hidden_dim=256,
            shared_feature_dim=128,
            shared_epochs=100,
            epochs=150,
            lr=0.001,
            batch_size=64
        )

        # assert False, y_train.shape
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
        num_parameters.append(n_parameters)
        print(f'----- Node hidden dimension = {d} -----')
        print(f'Train loss: {y_train_loss:.4f}, Train accuracy: {acc_train:.4f}')
        print(f'Test loss: {y_test_loss:.4f}, Test accuracy: {acc_test:.4f}')

    test_loss_sorted = [x for _, x in sorted(zip(num_parameters,test_loss))]
    min_test_loss = test_loss_sorted[0]; max_test_loss = test_loss_sorted[-1]
    test_losses = [(x-min_test_loss)/(max_test_loss-min_test_loss) for x in test_loss_sorted]

    train_loss_sorted = [x for _, x in sorted(zip(num_parameters,train_loss))]
    min_train_loss = train_loss_sorted[0]; max_train_loss = train_loss_sorted[-1]
    train_losses = [(x-min_train_loss)/(max_train_loss-min_train_loss) for x in train_loss_sorted]

    num_parameters.sort()
    parameters = [f'{(params/1000):.1f}K' for params in num_parameters]

    #double descent plot
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,8))
    ax.plot(num_parameters,train_losses,label='Train loss')
    ax.plot(num_parameters,test_losses,label='Test loss')
    ax.set_ylabel('Normalized loss')
    ax.set_xlabel('Tree network size (#parameters)')
    ax.set_xticks(num_parameters)
    ax.set_xticklabels(parameters)
    ax.legend()
    ax.grid(alpha=0.5)
    ax.tick_params(axis='x',labelrotation=45)
    fig.savefig(f'double_descent_{dataset}.png',bbox_inches='tight')
    plt.close(fig)
