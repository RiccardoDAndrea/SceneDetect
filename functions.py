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


#----------------------------------------------------------------------------------------------
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
#----------------------------------------------------------------------------------------------


def create_TrainDir(user_input_subDir):
    """
    
    """
    train_dir = "UploadedFile/Train"
    train_dir_class_1 = f"UploadedFile/Train/{user_input_subDir}"
    #train_dir_class_1 = f"UploadedFile/Train/{directory_name}"

    # Creates a directory based on the user input on which the classes are trained.
    os.makedirs(train_dir, exist_ok=True)

    if not os.path.exists(train_dir_class_1):
        os.makedirs(train_dir_class_1, exist_ok=True)
    return train_dir_class_1
    

#----------------------------------------------------------------------------------------------


def renameImages(image_files_1, train_dir_class_1):
    """
    
    """
    if image_files_1:
        # looping over the pictures to rename them in numbers (0,1,2).jepg
        for idx, image_file in enumerate(image_files_1):
            try:
                # Open and process image using Pillow
                img = Image.open(image_file)
                # checke if mode ist RGB otherwise create a error message
                # Keras function image_dataset_from_directory need RGB otherwise error message
                if img.mode == "RGBA":
                    img = img.convert("RGB")
                
                # Define unique save path
                save_path = os.path.join(train_dir_class_1, 
                                        f"{image_file.name}")
                img.save(save_path, format="PNG")
            
            except Exception as e:
                st.error(f"Fehler beim Verarbeiten von {image_file.name}: {e}")
    
    
        st.success(f"Uploaded {len(image_files_1)} training images.")
    else:
        st.warning("Please upload at least one training image.")

    return None

def checkDir_exist(train_dir_class_1):
    # Prüfen, ob der Ordner existiert
        if not os.path.exists(train_dir_class_1):
            st.error(f"The directory '{train_dir_class_1}' does not exist. Please create it and upload images.")
            st.stop()

        # Prüfen, ob der Ordner leer ist
        if not os.listdir(train_dir_class_1):
            st.error(f"The directory '{train_dir_class_1}' is empty. Please upload images.")
            st.stop()