import functions as fc
import os
import shutil
from PIL import Image
import streamlit as st
#----------------------------------------------------------------------------------------------
base_dir = "UploadedFile/Train"


def app():

    fc.create_directories(base_dir)

    ImageClassification_radio = st.radio(
                        "Image Classification",
                        ["One Way", "Two way"],
                        captions=[
                            "Recognize a picture.",
                            "Distinguish between two.",],
                        horizontal=True)

    
    ###########################
    ##### O N E _ W A Y #######
    ###########################

    if ImageClassification_radio == "One Way":
        # User Input names get used later for the Prediciton
        name_class_1 = st.text_input("Name Class 1", key="Name Class 1")
        train_dir_class_1=fc.create_TrainDir(user_input_subDir=name_class_1)
        
        
        if name_class_1: 
            image_files_1 = st.file_uploader(
                "Upload the first Class Images", 
                accept_multiple_files=True, 
                type=["jpg", "jpeg", "png","JPEG"], 
                key="file_uploader_1")
            
            
            fc.checkDir_exist(train_dir_class_1)
            fc.renameImages(image_files_1, train_dir_class_1)
            