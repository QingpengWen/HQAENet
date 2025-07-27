# -*- coding: utf-8 -*-
"""
@CreateTime :       2025/06/21 12:18
@File       :       cross_datasets.py
@Software   :       PyCharm
@Framework  :       Pytorch
@LastModify :       2025/07/30 23:35
"""
import os
import pickle
from modelscope.hub.snapshot_download import snapshot_download

def save_test_loader(loader, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(loader, f)
    print(f"Successfully save DataLoader to: {file_path}")

def load_test_loader(file_name):

    try:
        file_path = os.path.join(f"dataset/crossdatasets", file_name)
        with open(file_path, 'rb') as f:
            test_loader = pickle.load(f)
    except FileNotFoundError:
        download_name = os.path.join(f"crossdatasets", file_name)
        download_path = snapshot_download('Anony4Model/Parameters4HQAENet',
                                      allow_patterns=download_name,
                                      local_dir="dataset")
        file_path = os.path.join(download_path, download_name)
        with open(file_path, 'rb') as f:
            test_loader = pickle.load(f)

    # with open(file_path, 'rb') as f:
    #     test_loader = pickle.load(f)
    print(f"Successfully load TestLoader from {file_path}")
    return test_loader