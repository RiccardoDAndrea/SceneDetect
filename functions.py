import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import imageio.v2 as imageio 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.datasets import cifar100

def create_directories(base_dir): 
    """
    Create directorie for randomImages 

    Parameters
    ----------
    base_dir : string 
            Realtive Pathe to the Base dir    
    
    Returns
    -------
    None

    """
    exclude_dir = os.path.join(base_dir, "randomImages")
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            # Skip the "random Images" directory
            # “Random Images” is the directory of random images 
            # used to train a class model to tell if train_dir_class1 
            # is what the user enters on the image.
            if item_path == exclude_dir:
                continue
            # Delete files or directories
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)  
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path) 
    return None

create_directories(base_dir="UploadedFile/Train")


def create_TrainDir(directory_name):
    train_dir = "UploadedFile/Train"
    train_dir_class_1 = f"UploadedFile/Train/{name_class_1}"

    # Creates a directory based on the user input on which the classes are trained.
    os.makedirs(train_dir, exist_ok=True)

    if not os.path.exists(train_dir_class_1):
        os.makedirs(train_dir_class_1, exist_ok=True)