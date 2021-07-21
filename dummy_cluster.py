import os
import glob
import cv2
import argparse
from tqdm import tqdm

import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from tqdm import tqdm

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

def data_preprocess(data_path: str):
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

def dummy_cluster(embb_vec):
    """
    Using Kmeans to cluster embbeding vector from raw images
    """
    kmeans = KMeans(n_clusters=8, random_state=42)
    clusters = kmeans.fit_predict(embb_vec)
    return clusters

def get_cluster_class(all_labels, clusters):
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

def get_class(ref_classes, clusters):
    """
    Get actual class for each instances
    """
    pred_classes = np.zeros(len(clusters))
    for i in range(len(clusters)):
        pred_classes[i] = ref_classes[clusters[i]]
    return pred_classes

def main(args):
    flatten_images, all_labels = data_preprocess(args.data_path)
    clusters = dummy_cluster(embb_vec=flatten_images)
    ref_classes = get_cluster_class(all_labels, clusters)
    predicted = get_class(ref_classes, clusters)

    print(accuracy_score(all_labels, predicted))
    print(confusion_matrix(all_labels, predicted))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='',
                        help='Path to data directory')
    parser.add_argument('--exp_name', type=str, default='default',
                        help='Set experiment directory name')
    
    args = parser.parse_args()

    main(args)











