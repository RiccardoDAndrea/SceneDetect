import streamlit as st
import imageio.v2 as imageio 
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import pandas as pd
from tensorflow.keras.utils import image_dataset_from_directory
import os
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.datasets import cifar100
import os
from PIL import Image
import shutil

st.set_page_config(page_title="STATY AI", page_icon="🧊", layout="wide")

# Title
st.title("STATY AI")

# If user load webpage new, deleting the old dir
if os.path.exists("UploadedFile"):
    # Löscht den gesamten Ordner und seinen Inhalt
    shutil.rmtree("UploadedFile")


ImageClassification_radio = st.radio(
    "Image Classification",
    ["One Way", "Two way"],
    captions=[
        "Ein Bild erkennen.",
        "Zwei unterscheiden können.",],
    horizontal=True)


###########################
##### O N E _ W A Y #######
###########################
if ImageClassification_radio == "One Way":
    st.write("You selected comedy.")



###########################
##### T W O _ W A Y #######
###########################
elif ImageClassification_radio == "Two way":
    
     
    name_class1_col, name_class2_col = st.columns(2)
    with name_class1_col:
        # User Input names get used later for the Prediciton
        name_class_1 = st.text_input("Name Class 1", key="Name Class 1")

    with name_class2_col:
        # User Input names get used later for the Prediciton
        name_class_2 = st.text_input("Name Class 2", key="Name Class 2")   
    
    # Define directories for uploaded images
    if name_class_1 and name_class_2:   
        # The directory structure is crucial for image classification, 
        # as the 'image_dataset_from_directory' function requires the following hierarchy:
        # Root Directory
        #  └── Train
        #      └── Prediction Classes <- all images for the classification

        train_dir = "UploadedFile/Train"
        train_dir_class_1 = f"UploadedFile/Train/{name_class_1}"
        train_dir_class_2 = f"UploadedFile/Train/{name_class_2}"

        
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
                type=["jpg", "jpeg", "png","JPEG"], 
                key="file_uploader_1")

            if image_files_1:
                for idx, image_file in enumerate(image_files_1):
                    try:
                        # Open and process image using Pillow
                        img = Image.open(image_file)
                        if img.mode == "RGBA":
                            img = img.convert("RGB")
                        
                        # Define unique save path
                        save_path = os.path.join(train_dir_class_1, f"{idx}_{image_file.name}")
                        img.save(save_path, format="PNG")
                    
                    except Exception as e:
                        st.error(f"Fehler beim Verarbeiten von {image_file.name}: {e}")
            
            
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
                for idx, image_file in enumerate(image_files_2):
                    try:
                        # Open and process image using Pillow
                        img = Image.open(image_file)
                        if img.mode == "RGBA":
                            img = img.convert("RGB")
                        
                        # Define unique save path
                        save_path = os.path.join(train_dir_class_2, f"{idx}_{image_file.name}")
                        img.save(save_path, format="PNG")

                    except Exception as e:
                        st.error(f"Fehler beim Verarbeiten von {image_file.name}: {e}")  

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
                "Validation",
                image_size=(180, 180),
                batch_size=32
            )

    else:
        st.stop()

        


    #####################################################################

    # Model selection
    options = st.selectbox(
        "Which Models to test",
        ["Own Model","MobileNetV2", 
            "SENET", "ViT"],
    )
    st.divider()
    if "Own Model" in options:
        st.markdown("#### Create your infracture")
        
        # User input for the number of layers
        number_layers = st.number_input("Number of Layers", min_value=1, max_value=20, step=1)

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
                        min_value=1, max_value=1000, 
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
        #x = layers.Rescaling(1./255)(x)
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
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(unit, activation=activation)(x)  # Example output layer

        # Create the model
        model = keras.Model(inputs=inputs, outputs=outputs)
        st.divider()
        st.markdown("#### Compile Model")
        ###### User Interface for Compile Model
        
        loss_col, optimizer_col, epochs_col = st.columns(3)

        with loss_col:
            loss = st.selectbox(
                "Loss Function",
                ["binary_crossentropy", "categorical_crossentropy", "mean_squared_error"],
                key="loss"
            )
        
        with optimizer_col:
            optimizer = st.selectbox(
                "Optimizer",
                ["Adam", "rmsprop", "SGD"],
                key="optimizer"
            )

        with epochs_col:
            epochs_user = st.number_input("Epochs", min_value=1, max_value=100, step=1, key="epochs")
        
        # Initialize session state for training status
        if "training_completed" not in st.session_state:
            st.session_state.training_completed = False

        # Compile and train model
        # Reset training flag if user wants to train again
        if st.button("Reset Training Status"):
            st.session_state.training_completed = False
            st.info("Training status has been reset. You can train the model again.")

        # Button to compile and train the model
        if st.button("Compile and Train Model") and not st.session_state.training_completed:
            try:
                # Compile the model
                model.compile(loss=loss, optimizer=optimizer, 
                              metrics=["accuracy"])
                st.info("Model compilation successful.")
                
                # Train the model
                history = model.fit(
                    train_dataset,
                    epochs=epochs_user,
                    validation_data=validation_dataset
                )

                # Update session state
                st.session_state.model = model
                st.session_state.history = history.history 
                st.session_state.training_completed = True
                st.success("Model training completed successfully.")
                st.divider()

            except ValueError as e:
                error_message = str(e)
                if "Arguments `target` and `output` must have the same rank (ndim)" in error_message:
                    st.error(
                        "Shape mismatch detected: The output shape of the model does not match the target labels. "
                        "Ensure the output layer configuration matches the dataset labels. "
                        "For binary classification, use `Dense(1, activation='sigmoid')`."
                    )
                    st.info(
                        """
                        Your model should have the following configuration for binary classification:

                            - Input layer: `Input(shape=(180, 180, 3))`
                            - Conv2D with filters (32), kernel size (3), and activation (`relu`)
                            - MaxPooling2D with pool size (2)
                            - Conv2D with filters (64), kernel size (3), and activation (`relu`)
                            - MaxPooling2D with pool size (2)
                            - Conv2D with filters (128), kernel size (3), and activation (`relu`)
                            - Flatten layer
                            - Output layer: `Dense(1, activation='sigmoid')`
                        """
                    )
                else:
                    st.error(f"An unexpected error occurred: {error_message}")
                st.stop()

        data_eval = st.expander("Data Visulisation")
        with data_eval:
            if "history" in st.session_state:
                history_data = st.session_state.history  # Lade die gespeicherten History-Daten
                accuracy = history_data["accuracy"]
                val_accuracy = history_data["val_accuracy"]
                loss = history_data["loss"]
                val_loss = history_data["val_loss"]
                epochs_val = range(1, len(accuracy) + 1)

                # DataFrame erstellen
                data = {
                    'epochs': epochs_val,
                    'accuracy': accuracy, 
                    'val_accuracy': val_accuracy,
                    'loss': loss,
                    'val_loss': val_loss
                }

                df = pd.DataFrame(data)
                df = df.set_index('epochs')
                
                st.dataframe(df, use_container_width=True)
                st.divider()

                # Plot die Genauigkeit
                
                st.markdown("### Model Evaluation")
                st.line_chart(df[["accuracy", "val_accuracy"]], 
                            y_label=["accuracy", "val_accuracy"], 
                            use_container_width=True)
                
                st.line_chart(df[["loss", "val_loss"]], 
                            y_label=["accuracy", "val_accuracy"], 
                            use_container_width=True)
            
            else:
                st.info("Train the model first to visualize the results.")  












        if st.session_state.training_completed:
            st.divider()
            st.markdown("### Test your Model:")
            if "model" not in st.session_state:
                st.error("Model not found. Please train the model first.")
            else:
                uploaded_image = st.file_uploader("Upload an image to test your model", type=["png", "jpg", "jpeg"])

                if uploaded_image is not None:
                    img = image.load_img(uploaded_image, target_size=(180, 180))
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)

                    # Vorhersage des Modells
                    model = st.session_state.model
                    prediction_results = model.predict(img_array)

                    if prediction_results[0] > 0.5:
                        st.write(f"The image is classified as {name_class_1} with a probability of {prediction_results[0]}")
                    else:
                        st.write(f"The image is classified as {name_class_2} with a probability of {1 - prediction_results[0]}")
                else:
                    st.info("Please upload an image to test the model.")