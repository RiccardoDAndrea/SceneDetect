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

st.set_page_config(page_title="STATY AI", page_icon="🧊", layout="wide")

st.title("STATY AI")

import os
import shutil

base_dir = "UploadedFile/Train"
exclude_dir = os.path.join(base_dir, "randomImages")

if os.path.exists(base_dir):
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        # Überspringe das Verzeichnis "nothing"
        if item_path == exclude_dir:
            continue
        # Lösche Dateien oder Verzeichnisse
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)  # Löscht Dateien oder symbolische Links
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)  # Löscht ganze Verzeichnisse



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
    
    
    
        # User Input names get used later for the Prediciton
    name_class_1 = st.text_input("Name Class 1", key="Name Class 1")

    
    
    # Define directories for uploaded images
    if name_class_1:   
        # The directory structure is crucial for image classification, 
        # as the 'image_dataset_from_directory' function requires the following hierarchy:
        #---------------------------------------------------------------------------------
        # Root Directory
        #  └── Train
        #      └── Prediction Classes <- all images for the classification

        train_dir = "UploadedFile/Train"
        train_dir_class_1 = f"UploadedFile/Train/{name_class_1}"

        os.makedirs(train_dir, exist_ok=True)
        if not os.path.exists(train_dir_class_1):
            os.makedirs(train_dir_class_1, exist_ok=True)
            
        
        # File upload columns
         # Train Dataset Upload
        
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
                    # checke if mode ist RGB otherwise create a error message
                    if img.mode == "RGBA":
                        img = img.convert("RGB")
                    
                    # Define unique save path
                    save_path = os.path.join(train_dir_class_1, 
                                            f"{idx}_{image_file.name}")
                    img.save(save_path, format="PNG")
                
                except Exception as e:
                    st.error(f"Fehler beim Verarbeiten von {image_file.name}: {e}")
        
        
            st.success(f"Uploaded {len(image_files_1)} training images.")
        else:
            st.warning("Please upload at least one training image.")
        
        
        # Prüfen, ob der Ordner existiert
        if not os.path.exists(train_dir_class_1):
            st.error(f"Das Verzeichnis '{train_dir_class_1}' existiert nicht. Bitte erstelle es und lade Bilder hoch.")
            st.stop()

        # Prüfen, ob der Ordner leer ist
        if not os.listdir(train_dir_class_1):
            st.error(f"Das Verzeichnis '{train_dir_class_1}' ist leer. Bitte lade Bilder hoch.")
            st.stop()

        # Dataset laden
        train_dataset = image_dataset_from_directory(
            train_dir,
            image_size=(180, 180),
            batch_size=32
        )
        validation_dataset = image_dataset_from_directory(
            "cats_vs_dogs_small/validation",
            image_size=(180, 180),
            batch_size=32
        )

        st.success("Trainingsdaten erfolgreich geladen.")

    else:
        st.stop()


    st.divider()

    ##########################################
    ###### O W N _ I N F R A C T U R E #######
    ##########################################
    st.markdown("#### Create your infracture")
    
    # User input for the number of layers
    # max_value is sett to 20 but its possible to increase the number input
    number_layers = st.number_input("Number of Layers", min_value=1, max_value=20, step=1)

    # Initialize storage lists
    layer_types = []
    filters = []
    kernels = []
    MAXpool_sizes = []
    AVGpool_sizes = []
    units = []
    activations = []
    dropout_rates = []
    Spatialdropout_rates = []

    # Loop through each layer
    for i in range(number_layers):

        st.divider()
        ## Creating the User Interface
        
        # Select the layer type
        layer_type = st.selectbox(
            f'Layer {i+1} Type',
            ['Conv2D', 'MaxPooling2D', 'Flatten', 
             'BatchNormalization', 'AveragePooling2D','Dropout','SpatialDropout2D','Dense'],
            key=f'layer_type_{i}')
        layer_types.append(layer_type)

        # Create input columns for Conv2D layer
        if layer_type == "Conv2D":
            filter_col, kernel_size_col, activation_col = st.columns(3)

            with filter_col:
                filters.append(st.number_input(
                    f'Filters (Layer {i+1})',
                    min_value=1, max_value=1000,
                    value=32, step=1, key=f'filter_{i}'))

                

            with kernel_size_col:
                kernels.append(st.number_input(
                    f'Kernel Size (Layer {i+1})',
                    min_value=1, max_value=10,
                    value=3, step=1, key=f'kernel_size_{i}'))

            with activation_col:
                activations.append(st.selectbox(
                    f'Activation (Layer {i+1})',
                    ['relu', 'sigmoid', 'tanh'],
                    key=f'activation_{i}'))

            # Append placeholders for other lists
            units.append(None)
            MAXpool_sizes.append(None)
            AVGpool_sizes.append(None)
            Spatialdropout_rates.append(None)
            dropout_rates.append(None)   

        elif layer_type == "Dense":
            # Create input columns for Dense layer
            units_col, activation_col = st.columns(2)
            with units_col:
                units.append(st.number_input(
                    f'Units (Layer {i+1})',
                    min_value=1, max_value=512,
                    value=128, step=1, key=f'units_{i}'))


            with activation_col:
                activations.append(st.selectbox(
                    f'Activation (Layer {i+1})',
                    ['relu', 'sigmoid', 'tanh'],
                    key=f'activation_dense_{i}'))

                # Append placeholders for other lists
                filters.append(None)
                kernels.append(None)
                Spatialdropout_rates.append(None)
                MAXpool_sizes.append(None)
                AVGpool_sizes.append(None)
                dropout_rates.append(None)

        elif layer_type == "MaxPooling2D":
            MAXpool_sizes.append(st.number_input(
                f'Pool Size (Layer {i+1})',
                min_value=1, max_value=5,
                value=2, step=1, key=f'pool_size_{i}'))

            # Append placeholders for other lists
            filters.append(None)
            kernels.append(None)
            units.append(None)
            Spatialdropout_rates.append(None)
            activations.append(None)
            dropout_rates.append(None)
        
        elif layer_type == "AveragePooling2D":
            AVGpool_sizes.append(st.number_input(
                f'Pool Size (Layer {i+1})',
                min_value=1, max_value=5,
                value=2, step=1, key=f'pool_size_{i}'))

            # Append placeholders for other lists
            filters.append(None)
            kernels.append(None)
            units.append(None)
            Spatialdropout_rates.append(None)
            MAXpool_sizes.append(None)
            activations.append(None)
            dropout_rates.append(None)
        
        elif layer_type == "BatchNormalization":
            filters.append(None)
            kernels.append(None)
            units.append(None)
            activations.append(None)
            Spatialdropout_rates.append(None)
            AVGpool_sizes.append(None)
            MAXpool_sizes.append(None)
            dropout_rates.append(None)


        elif layer_type == "Dropout":
            dropout_rates.append(st.number_input(
                f'Dropout Rate (Layer {i+1})',
                min_value=0.0, max_value=0.5, step=0.1, key=f'dropout_{i}'))

            # Append placeholders for other lists
            filters.append(None)
            kernels.append(None)
            units.append(None)
            Spatialdropout_rates.append(None)
            activations.append(None)
            MAXpool_sizes.append(None)
            AVGpool_sizes.append(None)
        
        elif layer_type == "SpatialDropout2D":
            Spatialdropout_rates.append(st.number_input(
                f'SpatialDropout2D Rate (Layer {i+1})',
                min_value=0.0, max_value=0.5, step=0.1, key=f'dropout_{i}'))

            # Append placeholders for other lists
            filters.append(None)
            kernels.append(None)
            units.append(None)
            activations.append(None)
            MAXpool_sizes.append(None)
            AVGpool_sizes.append(None)

        elif layer_type == "Flatten":
            # Append placeholders for all lists
            filters.append(None)
            kernels.append(None)
            units.append(None)
            Spatialdropout_rates.append(None)
            activations.append(None)
            AVGpool_sizes.append(None)
            MAXpool_sizes.append(None)
            dropout_rates.append(None)
    
    # First Layer: Input layer
    inputs = keras.Input(shape=(180, 180, 3))
    
    # Data Augmentation layer for a more robust model for prediciton
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.2),
        ])

    x = data_augmentation(inputs)  # Start with augmented input
    x = layers.Rescaling(1./255)(x)   

    # Build layers dynamically
    for i in range(number_layers):
        layer_type = layer_types[i]
        if layer_type == "Conv2D":
            x = layers.Conv2D(filters=filters[i], kernel_size=kernels[i], activation=activations[i])(x)
        elif layer_type == "Dense":
            x = layers.Dense(units[i], activation=activations[i])(x)
        elif layer_type == "Dropout":
            x = layers.Dropout(rate=dropout_rates[i])(x)
        elif layer_type == "SpatialDropout2D":
            x = layers.SpatialDropout2D(rate=Spatialdropout_rates[i])(x)
        elif layer_type == "Flatten":
            x = layers.Flatten()(x)
        elif layer_type == "BatchNormalization":
            x = layers.BatchNormalization()(x)
        elif layer_type == "MaxPooling2D":
            x = layers.MaxPooling2D(pool_size=MAXpool_sizes[i])(x)
        elif layer_type == "AveragePooling2D":
            x = layers.AveragePooling2D(pool_size=AVGpool_sizes[i])(x)
        


    # Final output layer
    # TODO: Create dropout layer

    # if len(units) == 0:
    #     st.info("""
    #             You must define at least one layer type to build the CNN architecture.
    #             Your model should have the following configuration for binary classification:

    #                     - Input layer: Input(shape=(180, 180, 3))
    #                     - Conv2D with filters (32), kernel size (3), and activation (`relu`)
    #                     - MaxPooling2D with pool size (2)
    #                     - Conv2D with filters (64), kernel size (3), and activation (`relu`)
    #                     - MaxPooling2D with pool size (2)
    #                     - Conv2D with filters (128), kernel size (3), and activation (`relu`)
    #                     - Flatten layer
    #                     - Output layer: `Dense(1, activation='sigmoid')
    #             """)

        # st.stop()
    st.divider()
    st.subheader("Output Layer Configuration")

    output_units = st.number_input(
        "Number of Output Units", min_value=1, value=1, step=1, key="output_units")
    output_activation = st.selectbox(
        "Output Activation", ['sigmoid', 'softmax', 'linear'], key="output_activation")

    # Erstelle die Ausgabe-Schicht basierend auf Benutzereingaben
    outputs = layers.Dense(output_units, activation=output_activation)(x)

    # Modell erstellen
    model = keras.Model(inputs=inputs, outputs=outputs)
    # Create the model
    print("==========================================")
    print(model.summary())
    print("==========================================")

    st.divider()
    st.markdown("#### Compile Model")\
    
    #################################
    ### C O M P I L E _ M O D E L ###
    #################################
    loss_col, optimizer_col, epochs_col = st.columns(3)

    with loss_col:
        loss = st.selectbox(
            "Loss Function",
                ["binary_crossentropy", "categorical_crossentropy", "mean_squared_error"],
                    key="loss")
    
    with optimizer_col:
        optimizer = st.selectbox(
            "Optimizer",
                ["Adam", "rmsprop", "SGD"],
                    key="optimizer")

    with epochs_col:
        epochs_user = st.number_input("Epochs", min_value=1, 
                                    max_value=100, step=1, 
                                    key="epochs")
    
    # Initialize session state for training status and model storage
    # st.session_state is used to persist the model and training status across user interactions.
    # This ensures that the model remains available for predictions even after the page reloads,
    # for example, when a picture is uploaded for testing. Without this, the model would not be found,
    # as Streamlit re-runs the script on every interaction.

    if "training_completed" not in st.session_state:
        st.session_state.training_completed = False

    # COMPILE AND TRAIN MODEL
    # Reset training flag if user wants to train again
    ResetTraining_col,CompileTrain_col = st.columns(2)
    with ResetTraining_col:
        if st.button("Reset Training Status"):
            # Reset the variable training_completed so user can train a new CNN-Model
            st.session_state.training_completed = False
            st.info("Training status has been reset. You can train the model again.")
    with  CompileTrain_col:
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
                # SAVE all the variables for CNN Models to run and predict
                st.session_state.model = model
                st.session_state.history = history.history 
                st.session_state.training_completed = True
                st.success("Model training completed successfully.")
                

            except ValueError as e:
                # User get a error message because to create a CNN Infrastruce is crucial
                # and a lot can go wrong so the user get a example Infrastrucer.
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
                            - Output layer: `Dense(1, activation='sigmoid')
                        """
                    )
                else:
                    st.error(f"An unexpected error occurred: {error_message}")
                st.stop()

    #############################################
    #### D A T A _ V I S U A L I Z A T I O N ####
    #############################################

    data_eval = st.expander("Data Visualization")
    # Plotting the Model evaluation in a expander
    with data_eval:

        if "history" in st.session_state:
            history_data = st.session_state.history  # getting the model histort of the session_state
            accuracy = history_data["accuracy"]     
            val_accuracy = history_data["val_accuracy"]
            loss = history_data["loss"]
            val_loss = history_data["val_loss"]
            epochs_val = range(1, len(accuracy) + 1)

            # Create a Dataframe to Visualization for a line plot
            data = {
                'epochs': epochs_val,
                'accuracy': accuracy, 
                'val_accuracy': val_accuracy,
                'loss': loss,
                'val_loss': val_loss}

            df = pd.DataFrame(data)
            df = df.set_index('epochs')
            
            st.dataframe(df, use_container_width=True)
            st.divider()

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
            uploaded_image = st.file_uploader("Upload an image to test your model", 
                                                type=["png", "jpg", "jpeg"])

            if uploaded_image is not None:
                img = image.load_img(uploaded_image, target_size=(180, 180))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)

                # Vorhersage des Modells
                model = st.session_state.model
                prediction_results = model.predict(img_array)
                st.write(prediction_results)
                if prediction_results > 0.5:
                    st.markdown(f"The image is classified as {name_class_1} with a probability of {prediction_results[0]}")
                else:
                    st.write(f"Nothing classified {prediction_results[0][0]}")
            else:
                st.info("Please upload an image to test the model.")



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
        #---------------------------------------------------------------------------------
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
                        # checke if mode ist RGB otherwise create a error message
                        if img.mode == "RGBA":
                            img = img.convert("RGB")
                        
                        # Define unique save path
                        save_path = os.path.join(train_dir_class_1, 
                                                f"{idx}_{image_file.name}")
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
                        # checke if mode ist RGB otherwise create a error message
                        if img.mode == "RGBA":
                            img = img.convert("RGB")
                        
                        # Define unique save path
                        save_path = os.path.join(train_dir_class_2, 
                                                f"{idx}_{image_file.name}")
                        
                        img.save(save_path, format="PNG")

                    except Exception as e:
                        st.error(f"Fehler beim Verarbeiten von {image_file.name}: {e}")  

                st.success(f"Uploaded {len(image_files_2)} validation images.")
            else:
                st.warning("Please upload at least one validation image.")
                st.stop()


            # as mentioed before the dir strucutre is curcial for CNN in Tensorflow.Keras
            # image_size and batch_size ist changeble
            train_dataset = image_dataset_from_directory(
                train_dir,
                image_size=(180, 180),
                batch_size=32
            )
            
            validation_dataset = image_dataset_from_directory(
                "Validation",
                image_size=(180, 180),
                batch_size=32
            )

    else:
        st.stop()


    st.divider()

    ##########################################
    ###### O W N _ I N F R A C T U R E #######
    ##########################################
    st.markdown("#### Create your infracture")
    
    # User input for the number of layers
    # max_value is sett to 20 but its possible to increase the number input
    number_layers = st.number_input("Number of Layers", min_value=1, max_value=20, step=1)

    # Initialize storage lists
    layer_types = []
    filters = []
    kernels = []
    MAXpool_sizes = []
    AVGpool_sizes = []
    units = []
    activations = []
    dropout_rates = []
    Spatialdropout_rates = []

    # Loop through each layer
    for i in range(number_layers):

        st.divider()
        ## Creating the User Interface
        
        # Select the layer type
        layer_type = st.selectbox(
            f'Layer {i+1} Type',
            ['Conv2D', 'MaxPooling2D', 'Flatten', 
             'BatchNormalization', 'AveragePooling2D','Dropout','SpatialDropout2D','Dense'],
            key=f'layer_type_{i}')
        layer_types.append(layer_type)

        # Create input columns for Conv2D layer
        if layer_type == "Conv2D":
            filter_col, kernel_size_col, activation_col = st.columns(3)

            with filter_col:
                filters.append(st.number_input(
                    f'Filters (Layer {i+1})',
                    min_value=1, max_value=1000,
                    value=32, step=1, key=f'filter_{i}'))

                

            with kernel_size_col:
                kernels.append(st.number_input(
                    f'Kernel Size (Layer {i+1})',
                    min_value=1, max_value=10,
                    value=3, step=1, key=f'kernel_size_{i}'))

            with activation_col:
                activations.append(st.selectbox(
                    f'Activation (Layer {i+1})',
                    ['relu', 'sigmoid', 'tanh'],
                    key=f'activation_{i}'))

            # Append placeholders for other lists
            units.append(None)
            MAXpool_sizes.append(None)
            AVGpool_sizes.append(None)
            Spatialdropout_rates.append(None)
            dropout_rates.append(None)   

        elif layer_type == "Dense":
            # Create input columns for Dense layer
            units_col, activation_col = st.columns(2)
            with units_col:
                units.append(st.number_input(
                    f'Units (Layer {i+1})',
                    min_value=1, max_value=512,
                    value=128, step=1, key=f'units_{i}'))


            with activation_col:
                activations.append(st.selectbox(
                    f'Activation (Layer {i+1})',
                    ['relu', 'sigmoid', 'tanh'],
                    key=f'activation_dense_{i}'))

                # Append placeholders for other lists
                filters.append(None)
                kernels.append(None)
                Spatialdropout_rates.append(None)
                MAXpool_sizes.append(None)
                AVGpool_sizes.append(None)
                dropout_rates.append(None)

        elif layer_type == "MaxPooling2D":
            MAXpool_sizes.append(st.number_input(
                f'Pool Size (Layer {i+1})',
                min_value=1, max_value=5,
                value=2, step=1, key=f'pool_size_{i}'))

            # Append placeholders for other lists
            filters.append(None)
            kernels.append(None)
            units.append(None)
            Spatialdropout_rates.append(None)
            activations.append(None)
            AVGpool_sizes.append(None)
            dropout_rates.append(None)
        
        elif layer_type == "AveragePooling2D":
            AVGpool_sizes.append(st.number_input(
                f'Pool Size (Layer {i+1})',
                min_value=1, max_value=5,
                value=2, step=1, key=f'pool_size_{i}'))

            # Append placeholders for other lists
            filters.append(None)
            kernels.append(None)
            units.append(None)
            Spatialdropout_rates.append(None)
            MAXpool_sizes.append(None)
            activations.append(None)
            dropout_rates.append(None)
        
        elif layer_type == "BatchNormalization":
            filters.append(None)
            kernels.append(None)
            units.append(None)
            activations.append(None)
            Spatialdropout_rates.append(None)
            AVGpool_sizes.append(None)
            MAXpool_sizes.append(None)
            dropout_rates.append(None)


        elif layer_type == "Dropout":
            dropout_rates.append(st.number_input(
                f'Dropout Rate (Layer {i+1})',
                min_value=0.0, max_value=0.5, step=0.1, key=f'dropout_{i}'))

            # Append placeholders for other lists
            filters.append(None)
            kernels.append(None)
            units.append(None)
            Spatialdropout_rates.append(None)
            activations.append(None)
            MAXpool_sizes.append(None)
            AVGpool_sizes.append(None)
        
        elif layer_type == "SpatialDropout2D":
            Spatialdropout_rates.append(st.number_input(
                f'SpatialDropout2D Rate (Layer {i+1})',
                min_value=0.0, max_value=0.5, step=0.1, key=f'dropout_{i}'))

            # Append placeholders for other lists
            filters.append(None)
            kernels.append(None)
            units.append(None)
            activations.append(None)
            MAXpool_sizes.append(None)
            AVGpool_sizes.append(None)

        elif layer_type == "Flatten":
            # Append placeholders for all lists
            filters.append(None)
            kernels.append(None)
            units.append(None)
            Spatialdropout_rates.append(None)
            activations.append(None)
            AVGpool_sizes.append(None)
            MAXpool_sizes.append(None)
            dropout_rates.append(None)
    
    # First Layer: Input layer
    inputs = keras.Input(shape=(180, 180, 3))
    
    # Data Augmentation layer for a more robust model for prediciton
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.2),
        ])

    x = data_augmentation(inputs)  # Start with augmented input
    x = layers.Rescaling(1./255)(x)   

    # Build layers dynamically
    for i in range(number_layers):
        layer_type = layer_types[i]
        if layer_type == "Conv2D":
            x = layers.Conv2D(filters=filters[i], kernel_size=kernels[i], activation=activations[i])(x)
        elif layer_type == "Dense":
            x = layers.Dense(units[i], activation=activations[i])(x)
        elif layer_type == "Dropout":
            x = layers.Dropout(rate=dropout_rates[i])(x)
        elif layer_type == "SpatialDropout2D":
            x = layers.SpatialDropout2D(rate=Spatialdropout_rates[i])(x)
        elif layer_type == "Flatten":
            x = layers.Flatten()(x)
        elif layer_type == "BatchNormalization":
            x = layers.BatchNormalization()(x)
        elif layer_type == "MaxPooling2D":
            x = layers.MaxPooling2D(pool_size=MAXpool_sizes[i])(x)
        elif layer_type == "AveragePooling2D":
            x = layers.AveragePooling2D(pool_size=AVGpool_sizes[i])(x)
        


    # Final output layer
    # TODO: Create dropout layer

    # if len(units) == 0:
    #     st.info("""
    #             You must define at least one layer type to build the CNN architecture.
    #             Your model should have the following configuration for binary classification:

    #                     - Input layer: Input(shape=(180, 180, 3))
    #                     - Conv2D with filters (32), kernel size (3), and activation (`relu`)
    #                     - MaxPooling2D with pool size (2)
    #                     - Conv2D with filters (64), kernel size (3), and activation (`relu`)
    #                     - MaxPooling2D with pool size (2)
    #                     - Conv2D with filters (128), kernel size (3), and activation (`relu`)
    #                     - Flatten layer
    #                     - Output layer: `Dense(1, activation='sigmoid')
    #             """)

        # st.stop()
    st.divider()
    st.subheader("Output Layer Configuration")

    output_units = st.number_input(
        "Number of Output Units", min_value=1, value=1, step=1, key="output_units")
    output_activation = st.selectbox(
        "Output Activation", ['sigmoid', 'softmax', 'linear'], key="output_activation")

    # Erstelle die Ausgabe-Schicht basierend auf Benutzereingaben
    outputs = layers.Dense(output_units, activation=output_activation)(x)

    # Modell erstellen
    model = keras.Model(inputs=inputs, outputs=outputs)
    # Create the model
    print("==========================================")
    print(model.summary())
    print("==========================================")

    st.divider()
    st.markdown("#### Compile Model")\
    
    #################################
    ### C O M P I L E _ M O D E L ###
    #################################
    loss_col, optimizer_col, epochs_col = st.columns(3)

    with loss_col:
        loss = st.selectbox(
            "Loss Function",
            ["binary_crossentropy", "categorical_crossentropy", "mean_squared_error"],
            key="loss")
    
    with optimizer_col:
        optimizer = st.selectbox(
            "Optimizer",
                ["Adam", "rmsprop", "SGD"],
                    key="optimizer")

    with epochs_col:
        epochs_user = st.number_input("Epochs", min_value=1, 
                                    max_value=100, step=1, 
                                    key="epochs")
    
    # Initialize session state for training status and model storage
    # st.session_state is used to persist the model and training status across user interactions.
    # This ensures that the model remains available for predictions even after the page reloads,
    # for example, when a picture is uploaded for testing. Without this, the model would not be found,
    # as Streamlit re-runs the script on every interaction.

    if "training_completed" not in st.session_state:
        st.session_state.training_completed = False

    # COMPILE AND TRAIN MODEL
    # Reset training flag if user wants to train again
    ResetTraining_col,CompileTrain_col = st.columns(2)

    with ResetTraining_col:
        if st.button("Reset Training Status"):
            # Reset the variable training_completed so user can train a new CNN-Model
            st.session_state.training_completed = False
            st.info("Training status has been reset. You can train the model again.")
    with  CompileTrain_col:
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
                # SAVE all the variables for CNN Models to run and predict
                st.session_state.model = model
                st.session_state.history = history.history 
                st.session_state.training_completed = True
                st.success("Model training completed successfully.")
                

            except ValueError as e:
                # User get a error message because to create a CNN Infrastruce is crucial
                # and a lot can go wrong so the user get a example Infrastrucer.
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
                            - Output layer: `Dense(1, activation='sigmoid')
                        """
                    )
                else:
                    st.error(f"An unexpected error occurred: {error_message}")
                st.stop()

    #############################################
    #### D A T A _ V I S U A L I Z A T I O N ####
    #############################################

    data_eval = st.expander("Data Visualization")
    # Plotting the Model evaluation in a expander
    with data_eval:

        if "history" in st.session_state:
            history_data = st.session_state.history  # getting the model histort of the session_state
            accuracy = history_data["accuracy"]     
            val_accuracy = history_data["val_accuracy"]
            loss = history_data["loss"]
            val_loss = history_data["val_loss"]
            epochs_val = range(1, len(accuracy) + 1)

            # Create a Dataframe to Visualization for a line plot
            data = {
                'epochs': epochs_val,
                'accuracy': accuracy, 
                'val_accuracy': val_accuracy,
                'loss': loss,
                'val_loss': val_loss}

            df = pd.DataFrame(data)
            df = df.set_index('epochs')
            
            st.dataframe(df, use_container_width=True)
            st.divider()

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
            uploaded_image = st.file_uploader("Upload an image to test your model", 
                                                type=["png", "jpg", "jpeg"])

            if uploaded_image is not None:
                img = image.load_img(uploaded_image, target_size=(180, 180))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)

                # Vorhersage des Modells
                model = st.session_state.model
                prediction_results = model.predict(img_array)
                st.write(prediction_results)
                if prediction_results > 0.5:
                    st.markdown(f"The image is classified as {name_class_1} with a probability of {prediction_results[0]}")
                else:
                    st.write(f"The image is classified as {name_class_2} with a probability of {1 - prediction_results[0]}")
            else:
                st.info("Please upload an image to test the model.")