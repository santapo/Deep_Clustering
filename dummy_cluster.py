import os
import sys
import glob
import cv2
import argparse
import logging
from tqdm import tqdm

from typing import List

import numpy as np
from matplotlib import pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torchvision.models as models

from tensorboardX import SummaryWriter

logger = logging.getLogger()

CLASS_ID = {
    'butterfly' : 0,
    'cat'       : 1,
    'dog'       : 2,
    'forest'    : 3,
    'lion'      : 4,
    'monkey'    : 5,
    'rose'      : 6,
    'sunflower' : 7
    }

MEAN = np.array([0.5, 0.5, 0.5],
                   dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.5, 0.5, 0.5],
                dtype=np.float32).reshape(1, 1, 3)


def norm_input(image):
    """
    Normalizing inputs and covert it to torch tensor
    """
    image = (image.astype(np.float32) / 255.)
    image = (image - MEAN) / STD
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image,0)
    image = torch.from_numpy(image)
    return image

def get_backbone(model_name):
    if model_name == 'Resnet50':
        backbone = models.resnet50(pretrained=True)

    else:
        raise ValueError('This model name {} does not defined'.format(model_name))
    
    modules = list(backbone.children())[:-1]
    model = nn.Sequential(*modules)
    for p in model.parameters():
        p.requires_grad = False

    return model

def dummy_data_preprocess(data_path: str):
    """
    Load and preprocess all dataset's samples
    """
    all_images = []
    all_labels = []
    
    all_samples = glob.glob(os.path.join(data_path, '*/*.jpg'))
    for idx in tqdm(range(len(all_samples))):
        sample = all_samples[idx]
        image = cv2.imread(sample)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        all_images.append(image)

        label = sample.split('/')[-2]
        label_id = CLASS_ID.get(label)
        all_labels.append(label_id)

    all_images = np.array(all_images)
    all_labels = np.array(all_labels)

    # normalize and flatten image shape
    all_images = all_images / 255.
    flatten_images = all_images.reshape(len(all_images), -1)

    return flatten_images, all_labels

def resnet_data_preprocess(data_path: str):
    """
    Load and get features from pretrained resnet
    """
    model = get_backbone('Resnet50')

    all_images = []
    all_labels = []
    
    all_samples = glob.glob(os.path.join(data_path, '*/*.jpg'))
    for idx in tqdm(range(len(all_samples))):
        sample = all_samples[idx]
        image = cv2.imread(sample)
        image = cv2.resize(image, (224, 224))
        image = norm_input(image)
        # import ipdb; ipdb.set_trace()

        feature = model(image).flatten().numpy()
        all_images.append(feature)

        label = sample.split('/')[-2]
        label_id = CLASS_ID.get(label)
        all_labels.append(label_id)

    all_images = np.array(all_images)
    all_labels = np.array(all_labels)

    # flatten image shape
    import ipdb; ipdb.set_trace()
    flatten_images = all_images.reshape(len(all_images), -1)

    return flatten_images, all_labels

def dummy_cluster(embb_vec: np.ndarray,
                num_clusters: int):
    """
    Using Kmeans to cluster embbeding vector from raw images
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, verbose=0)
    clusters = kmeans.fit_predict(embb_vec)
    sum_squared = kmeans.inertia_
    return clusters, sum_squared

def get_cluster_class(all_labels: np.ndarray,
                        clusters: np.ndarray) -> np.ndarray:
    """
    Get the class that refer to each cluster.
    Class that have the most instances in a cluster will be
    assign as cluster's class reference. 
    """
    ref_classes = {}
    for i in range(len(np.unique(clusters))):
        cluster_idx = np.where(clusters == i,1,0)
        cluster_cls = np.bincount(all_labels[cluster_idx==1]).argmax()
        ref_classes[i] = cluster_cls
    return ref_classes

def get_class(ref_classes: np.ndarray,
                clusters: np.ndarray) -> np.ndarray:
    """
    Get actual class for each instances
    """
    pred_classes = np.zeros(len(clusters))
    for i in range(len(clusters)):
        pred_classes[i] = ref_classes[clusters[i]]
    return pred_classes

def plot_fig(confusion_matrix,
            class_name):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion_matrix, interpolation='nearest')

    ax.set_xticklabels(['']+class_name)
    ax.set_yticklabels(['']+class_name)
    fig.colorbar(cax)

    return fig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='',
                        help='Path to data directory')
    parser.add_argument('--exp_name', type=str, default='default',
                        help='Set experiment directory name')
    parser.add_argument('--min_cluster', type=int, default=8,
                        help='Min number of clusters')
    parser.add_argument('--max_cluster', type=int, default=10,
                        help='Max number of clusters')
    parser.add_argument('--use_feat_extractor', action='store_true',
                        help='Use feature extractor')
    
    args = parser.parse_args()
    
    # Setting up logging tools
    exp_dir = os.path.join(os.getcwd(), 'exps', args.exp_name)
    writer = SummaryWriter(exp_dir)
    
    # Get embedding vector
    class_name = list(CLASS_ID)
    if args.use_feat_extractor:
        flatten_images, all_labels = resnet_data_preprocess(args.data_path)
    else:
        flatten_images, all_labels = dummy_data_preprocess(args.data_path)

    # Run KMeans
    for n in range(args.min_cluster, args.max_cluster + 1):
        print(f'KMeans with {n} clusters')
        clusters, sum_squared = dummy_cluster(embb_vec=flatten_images, num_clusters=n)
        ref_classes = get_cluster_class(all_labels, clusters)
        predicted = get_class(ref_classes, clusters)
        
        acc = accuracy_score(all_labels, predicted)
        cm = confusion_matrix(all_labels, predicted)
        cm_fig = plot_fig(cm, class_name)

        writer.add_scalar('Accuracy', acc, n)
        writer.add_scalar('Elbow', sum_squared, n)
        writer.add_figure(tag='Confusion Matrix', figure=cm_fig, global_step=n)









