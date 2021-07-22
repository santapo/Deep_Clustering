Dataset:
- no_label: for clustering
- label: test clustering

TODO:
- Using Kmeans to cluster raw images
- Using Kmeans to cluster embbeding vector that from a backbone network
- Using Deep Clustering to cluster

Goal:
- Compare clustering result from three methods

Reference: 
https://www.youtube.com/watch?v=0m5GNDo-CFM

Evaluation Metric:
- Accuracy for each kmeans configuration
- Confusion
- Elbow

**1. Dummy Clustering**
```
usage: dummy_cluster.py [-h] [--data_path DATA_PATH] [--exp_name EXP_NAME] [--min_cluster MIN_CLUSTER] [--max_cluster MAX_CLUSTER] [--use_feat_extractor]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to data directory
  --exp_name EXP_NAME   Set experiment directory name
  --min_cluster MIN_CLUSTER
                        Min number of clusters
  --max_cluster MAX_CLUSTER
                        Max number of clusters
  --use_feat_extractor  Use feature extractor
```