import streamlit as st
import imageio.v2 as imageio 
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
import os

st.set_page_config(page_title="STATY AI", page_icon="🧊", layout="wide", initial_sidebar_state="expanded")

# Title
st.title("STATY AI")


ImageClassification_radio = st.radio(
    "Image Classification",
    ["One Way", "Two way"],
    captions=[
        "Ein Bild erkennen.",
        "Zwei unterscheiden können.",
    ],
    horizontal=True
)

if ImageClassification_radio == "One Way":
    st.write("You selected comedy.")

elif ImageClassification_radio == "Two way":

    # Define directories for uploaded images
    train_dir = "UploadedFile/Train"
    train_dir_class_1 = "UploadedFile/Train/class_1"
    train_dir_class_2 = "UploadedFile/Train/class_2"


    os.makedirs(train_dir, exist_ok=True)
    if not os.path.exists(train_dir_class_1):
        os.makedirs(train_dir_class_1, exist_ok=True)
        os.makedirs(train_dir_class_2, exist_ok=True)

    # File upload columns
    file_uploader_col_1, file_uploader_col_2 = st.columns(2)

    # Train Dataset Upload
    with file_uploader_col_1:
        image_files_1 = st.file_uploader(
            "Upload the first Class Images", 
            accept_multiple_files=True, 
            type=["jpg", "jpeg", "png"], 
            key="file_uploader_1")

        if image_files_1:
            for image_file in image_files_1:
                # Read and save the file to the training directory
                img = imageio.imread(image_file)
                save_path = os.path.join(train_dir_class_1, image_file.name)
                imageio.imsave(save_path, img)

        
            st.success(f"Uploaded {len(image_files_1)} training images.")
        else:
            st.warning("Please upload at least one training image.")
            

    with file_uploader_col_2:
        image_files_2 = st.file_uploader(
            "Upload the second Class Images", 
            accept_multiple_files=True, 
            type=["jpg", "jpeg", "png"], 
            key="file_uploader_2")

        if image_files_2:
            for image_file in image_files_2:
                # Read and save the file to the validation directory
                img = imageio.imread(image_file)
                save_path = os.path.join(train_dir_class_2, image_file.name)
                imageio.imsave(save_path, img)

            st.success(f"Uploaded {len(image_files_2)} validation images.")
        else:
            st.warning("Please upload at least one validation image.")
            st.stop()



        train_dataset = image_dataset_from_directory(
            train_dir,
            image_size=(180, 180),
            batch_size=32
        )
        print(train_dataset)
        
        validation_dataset = image_dataset_from_directory(
            "Validation_data",
            image_size=(180, 180),
            batch_size=32
        )

    


    #####################################################################

    # Model selection
    options = st.selectbox(
        "Which Models to test",
        ["Own Model","MobileNetV2", 
            "SENET", "ViT"],
    )
    st.divider()
    if "Own Model" in options:
        st.write("Own Model")
        
        # User input for the number of layers
        number_layers = st.number_input("Number of Layers", min_value=1, max_value=10, step=1)

        # Initialize storage lists
        layer_types = []
        filters = []
        kernels = []
        pool_sizes = []
        units = []
        activations = []

        # Loop through each layer
        for i in range(number_layers):
            st.divider()
            # Select the layer type
            layer_type = st.selectbox(
                f'Layer {i+1} Type', 
                ['Dense', 'Conv2D', 'MaxPooling2D', 'Flatten'], 
                key=f'layer_type_{i}'
            )

            layer_types.append(layer_type)

            if layer_type == "Conv2D":
                # Create input columns for Conv2D layer
                filter_col, kernel_size_col, activation_col = st.columns(3)
                with filter_col:
                    filter = st.number_input(
                        f'Filters (Layer {i+1})', 
                        min_value=1, max_value=64, 
                        value=32, step=1, key=f'filter_{i}'
                    )
                    filters.append(filter)

                with kernel_size_col:
                    kernel_size = st.number_input(
                        f'Kernel Size (Layer {i+1})', 
                        min_value=1, max_value=10, 
                        value=3, step=1, key=f'kernel_size_{i}'
                    )
                    kernels.append(kernel_size)

                with activation_col:
                    activation = st.selectbox(
                        f'Activation (Layer {i+1})', 
                        ['relu', 'sigmoid', 'tanh'], 
                        key=f'activation_{i}'
                    )
                    activations.append(activation)

            elif layer_type == "Dense":
                # Create input columns for Dense layer
                units_col, activation_col = st.columns(2)
                with units_col:
                    unit = st.number_input(
                        f'Units (Layer {i+1})', 
                        min_value=1, max_value=512, 
                        value=128, step=1, key=f'units_{i}'
                    )
                    units.append(unit)

                with activation_col:
                    activation = st.selectbox(
                        f'Activation (Layer {i+1})', 
                        ['relu', 'sigmoid', 'tanh'], 
                        key=f'activation_dense_{i}'
                    )
                    activations.append(activation)

            elif layer_type == "MaxPooling2D":
                # Create input for Maxpooling layer
                pool_size_col, _ = st.columns(2)
                with pool_size_col:
                    pool_size = st.number_input(
                        f'Pool Size (Layer {i+1})', 
                        min_value=1, max_value=5, 
                        value=2, step=1, key=f'pool_size_{i}'
                    )
                    pool_sizes.append(pool_size)

            elif layer_type == "Flatten":
                st.write("No additional parameters needed for Flatten layer.")

        

        # Input layer
        inputs = keras.Input(shape=(180, 180, 3))

        # Data Augmentation layer
        data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.2),
            ]
        )
        x = data_augmentation(inputs)  # Start with augmented input

        # Build layers dynamically
        for i in range(number_layers):
            layer_type = layer_types[i]
            
            if layer_type == "Conv2D":
                if i < len(filters) and i < len(kernels) and i < len(activations):
                    x = layers.Conv2D(filters=filters[i], kernel_size=kernels[i], activation=activations[i])(x)
                #else:
                #    st.error(f"Missing configuration for Conv2D layer at position {i+1}. Check your input.")
            
            elif layer_type == "Dense":
                if i < len(units) and i < len(activations):
                    x = layers.Dense(units[i], activation=activations[i])(x)
                #else:
                #    st.error(f"Missing configuration for Dense layer at position {i+1}. Check your input.")
            
            elif layer_type == "MaxPooling2D":
                if i < len(pool_sizes):
                    x = layers.MaxPooling2D(pool_size=pool_sizes[i])(x)
                #else:
                #    st.error(f"Missing pool size for Maxpooling layer at position {i+1}. Check your input.")
            
            elif layer_type == "Flatten":
                x = layers.Flatten()(x)

        # Final output layer
        outputs = layers.Dense(1, activation="sigmoid")(x)  # Example output layer

        # Create the model
        model = keras.Model(inputs=inputs, outputs=outputs)


        model.compile(loss="binary_crossentropy",
                optimizer="rmsprop",
                metrics=["accuracy"])
        
        with tf.device('/GPU:0'):
            history = model.fit(
            train_dataset,
            epochs=3,
            validation_data=validation_dataset)


        # st.write("Model Summary")
        # st.write("Model Training")
        # st.write("Model Evaluation")
        # st.write("Model Prediction")
        # st.write("Model Visualization")



    

