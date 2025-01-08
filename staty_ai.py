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


if os.path.exists('UploadedFile'):
    shutil.rmtree('UploadedFile')
base_dir = "UploadedFile/Train"

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

    
    # Define directories for uploaded images
    if name_class_1:   
        # The directory structure is crucial for image classification, 
        # as the 'image_dataset_from_directory' function requires the following hierarchy:
        #---------------------------------------------------------------------------------
        # Root Directory
        #  └── Train
        #      └── Prediction Classes <- all images for the classification
        # Both Pytorch and Tensorflow/Keras need an order structure for the ImageClassification to recognize which images (classes) are there.
        # If the folder is named Cat, all images in this folder are used to train the CNN model on the Cat.
        # Exactly the same if the other folder is called Dog     

        train_dir = "UploadedFile/Train"
        test_dir = "UploadedFile/Test"
        validation_dir = "UploadedFile/Validation"

        train_subDir_class_1 = f"UploadedFile/Train/{name_class_1}"
        test_subDir_class_1 = f"UploadedFile/Test/{name_class_1}"
        valid_subDir_class_1 = f"UploadedFile/Validation/{name_class_1}"
        
        # Creates a directory based on the user input on which the classes are trained.
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(validation_dir, exist_ok=True)
        
        if not os.path.exists(train_subDir_class_1):
            os.makedirs(train_subDir_class_1, exist_ok=True)
            os.makedirs(test_subDir_class_1, exist_ok=True)
            os.makedirs(valid_subDir_class_1, exist_ok=True)

        
        image_files_1 = st.file_uploader(
            "Upload the first Class Images", 
            accept_multiple_files=True, 
            type=["jpg", "jpeg", "png","JPEG"], 
            key="file_uploader_1")

        images_int = len(image_files_1)
        
        legth_dirTrain = int(images_int * .60)
        legth_dirTest = int(images_int * .20)
        legth_dirValid = int(images_int * .20)
        
        if image_files_1:
            # looping over the pictures to rename them in numbers (0,1,2).jepg
            for idx, image_file in enumerate(image_files_1):
                try:
                    # Open and process image using Pillow
                    img = Image.open(image_file)
                    # checke if mode ist RGB otherwise create a error message
                    # Keras function image_dataset_from_directory need RGB otherwise error message
                    # if img.mode == "RGBA":
                    #     img = img.convert("RGB")
                    
                    # Determine save path based on index
                    if idx < legth_dirTrain:
                        save_dir = train_subDir_class_1
                    elif idx < legth_dirTrain + legth_dirTest:
                        save_dir = test_subDir_class_1
                    else:
                        save_dir = valid_subDir_class_1

                    save_path = os.path.join(save_dir, f"{os.path.basename(image_file.name)}")
                    img.save(save_path, format="PNG")
                
                except Exception as e:
                    st.error(f"Error processing {image_file.name}: {e}")
        
        
        # Prüfen, ob der Ordner existiert
        directories_to_check = [
            f"UploadedFile/Test/{name_class_1}",
            f"UploadedFile/Train/{name_class_1}",
            f"UploadedFile/Validation/{name_class_1}"
        ]

        # Überprüfung aller Verzeichnisse
        for dir_path in directories_to_check:
            if not os.path.exists(dir_path):
                st.error(f"The directory ‘{dir_path}’ does not exist. Please create it and upload the images.")
                st.stop()
            if len(os.listdir(f"UploadedFile/Test/{name_class_1}")) == 0:
                st.warning("Directory is empty, pls upload Images")
                st.stop()

        # load Dataset 
        train_dataset = image_dataset_from_directory(
            train_dir,
            image_size=(180, 180),
            batch_size=32
        )

        test_dataset = image_dataset_from_directory(
            test_dir,
            image_size=(180, 180),
            batch_size=32
        )

        validation_dataset = image_dataset_from_directory(
            validation_dir,
            image_size=(180, 180),
            batch_size=32
        )

        st.success(f"Successful loading of images {images_int}")
    else:
        st.stop()


    st.divider()

    ##########################################
    ###### O W N _ I N F R A C T U R E #######
    ##########################################
    # Data Augmentation Expander mit Aktivierung/Deaktivierung
    data_augmentation_expander = st.expander("Data Augmentation")
    use_data_augmentation = st.checkbox("Enable Data Augmentation", value=False)

    if use_data_augmentation:
        with data_augmentation_expander:
            st.write("Configure data augmentation layers")

            augmentation_layers = st.number_input(
                "Number of Augmentation Layers", 
                min_value=1, max_value=5, step=1, 
                key="augmentation_layers"
            )

            augmentation_types = []  # Liste für die Augmentierungsschichten
            augmentation_params = []  # Liste für Parameter (falls nötig)

            for i in range(augmentation_layers):
                st.divider()
                st.subheader(f"Augmentation Layer {i+1}")
                augmentation_type = st.selectbox(
                    f'Layer {i+1} Type',
                    [
                        'RandomFlip', 'RandomRotation', 'RandomZoom',
                        'RandomTranslation', 'RandomContrast', 'RandomCrop'
                    ],
                    key=f'augmentation_type_{i}'
                )
                augmentation_types.append(augmentation_type)

                # Zusätzliche Parameter basierend auf dem Typ sammeln
                if augmentation_type == "RandomFlip":
                    flip_mode = st.selectbox(
                        f"Flip Mode (Layer {i+1})", 
                        ["horizontal", "vertical", "horizontal_and_vertical"], 
                        key=f'flip_mode_{i}'
                    )
                    augmentation_params.append({"mode": flip_mode})

                elif augmentation_type == "RandomRotation":
                    rotation_factor = st.slider(
                        f"Rotation Factor (Layer {i+1})", 
                        min_value=0.0, max_value=1.0, value=0.1, step=0.01, 
                        key=f'rotation_factor_{i}'
                    )
                    augmentation_params.append({"factor": rotation_factor})

                elif augmentation_type == "RandomZoom":
                    zoom_factor = st.slider(
                        f"Zoom Factor (Layer {i+1})", 
                        min_value=0.0, max_value=0.5, value=0.2, step=0.01, 
                        key=f'zoom_factor_{i}'
                    )
                    augmentation_params.append({"factor": zoom_factor})

                elif augmentation_type == "RandomTranslation":
                    height_factor = st.slider(
                        f"Height Factor (Layer {i+1})", 
                        min_value=0.0, max_value=0.5, value=0.1, step=0.01, 
                        key=f'height_factor_{i}'
                    )
                    width_factor = st.slider(
                        f"Width Factor (Layer {i+1})", 
                        min_value=0.0, max_value=0.5, value=0.1, step=0.01, 
                        key=f'width_factor_{i}'
                    )
                    augmentation_params.append({"height_factor": height_factor, "width_factor": width_factor})

                elif augmentation_type == "RandomContrast":
                    contrast_factor = st.slider(
                        f"Contrast Factor (Layer {i+1})", 
                        min_value=0.0, max_value=0.5, value=0.2, step=0.01, 
                        key=f'contrast_factor_{i}'
                    )
                    augmentation_params.append({"factor": contrast_factor})

                elif augmentation_type == "RandomCrop":
                    height = st.number_input(
                        f"Crop Height (Layer {i+1})", 
                        min_value=1, max_value=180, value=150, step=1, 
                        key=f'crop_height_{i}'
                    )
                    width = st.number_input(
                        f"Crop Width (Layer {i+1})", 
                        min_value=1, max_value=180, value=150, step=1, 
                        key=f'crop_width_{i}'
                    )
                    augmentation_params.append({"height": height, "width": width})

            # Dynamische Augmentierungsschichten erstellen
            st.subheader("Generated Augmentation Layers")
            augmentation_layers_list = []  # Liste der Augmentierungsschichten

            for i, aug_type in enumerate(augmentation_types):
                params = augmentation_params[i]

                if aug_type == "RandomFlip":
                    augmentation_layers_list.append(layers.RandomFlip(mode=params["mode"]))
                elif aug_type == "RandomRotation":
                    augmentation_layers_list.append(layers.RandomRotation(factor=params["factor"]))
                elif aug_type == "RandomZoom":
                    augmentation_layers_list.append(layers.RandomZoom(height_factor=params["factor"]))
                elif aug_type == "RandomTranslation":
                    augmentation_layers_list.append(layers.RandomTranslation(
                        height_factor=params["height_factor"], 
                        width_factor=params["width_factor"]
                    ))
                elif aug_type == "RandomContrast":
                    augmentation_layers_list.append(layers.RandomContrast(factor=params["factor"]))
                elif aug_type == "RandomCrop":
                    augmentation_layers_list.append(layers.RandomCrop(
                        height=params["height"], 
                        width=params["width"]
                    ))

            # Erstelle die data_augmentation Sequential-Schicht
            data_augmentation = keras.Sequential(augmentation_layers_list)
            st.write("Data Augmentation Configured:", data_augmentation)
    else:
        st.write("Data Augmentation is disabled.")
        data_augmentation = None


    st.markdown("#### Create your infracture")
    
    # User input for the number of layers
    # max_value is set to 20 but its possible to increase the number input
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
            [
             'Conv2D', 'MaxPooling2D', 'Flatten', 
             'BatchNormalization', 'AveragePooling2D','Dropout',
             'SpatialDropout2D','Dense'
             ],
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
            # To aviod the error messages "list out of index" we will append None to the 
            # untouches list to make the list the same length.
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
            MAXpool_sizes.append(
                    st.number_input(
                    f'Pool Size (Layer {i+1})',
                    min_value=1, max_value=5,
                    value=2, step=1, key=f'pool_size_{i}')
                    )

            # Append placeholders for other lists
            filters.append(None)
            kernels.append(None)
            units.append(None)
            Spatialdropout_rates.append(None)
            activations.append(None)
            dropout_rates.append(None)
        
        elif layer_type == "AveragePooling2D":
            AVGpool_sizes.append(
                    st.number_input(
                    f'Pool Size (Layer {i+1})',
                    min_value=1, max_value=5,
                    value=2, step=1, key=f'pool_size_{i}')
                    )

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
            dropout_rates.append(
                    st.number_input(
                    f'Dropout Rate (Layer {i+1})',
                    min_value=0.0, max_value=0.5, step=0.1, key=f'dropout_{i}')
                    )

            # Append placeholders for other lists
            filters.append(None)
            kernels.append(None)
            units.append(None)
            Spatialdropout_rates.append(None)
            activations.append(None)
            MAXpool_sizes.append(None)
            AVGpool_sizes.append(None)
        
        elif layer_type == "SpatialDropout2D":
            Spatialdropout_rates.append(
                    st.number_input(
                    f'SpatialDropout2D Rate (Layer {i+1})',
                    min_value=0.0, max_value=0.5, step=0.1, key=f'dropout_{i}')
                )

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
    # TODO: Adding more the increase Validaiton of the Model for better performance 
    

    if data_augmentation:
        x = data_augmentation(inputs)  # Verwende augmentierte Eingabe
    else:
        x = inputs  # Nutze rohe Eingabe
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

    # Create the output layer based on user input
    outputs = layers.Dense(output_units, activation=output_activation)(x)

    # Create Model
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Create the model overview for Dev if the layers get properly added
    print("==========================================")
    print(model.summary())
    print("==========================================")

    st.divider()
    st.markdown("#### Compile Model")
    
    #################################
    ### C O M P I L E _ M O D E L ###
    #################################
    loss_col, optimizer_col, epochs_col = st.columns(3)

    with loss_col:
        loss = st.selectbox(
            "Loss Function",
            [
            "binary_crossentropy", 
            "categorical_crossentropy", 
            "mean_squared_error"
            ],
            key="loss")
    
    with optimizer_col:
        optimizer = st.selectbox(
            "Optimizer",
                ["Adam", "rmsprop", "SGD"],
                    key="optimizer")

    with epochs_col:
        epochs_user = st.number_input(
                                    "Epochs", min_value=1, 
                                    max_value=100, step=1, 
                                    key="epochs"
                                    )
    
    # Initialize session state for training status and model storage
    # st.session_state is used to persist the model and training status across user interactions.
    # This ensures that the model remains available for predictions even after the page reloads,
    # for example, when a picture is uploaded for testing. Without this, the model would 
    # not be found, as Streamlit re-runs the script on every interaction.

    if "training_completed" not in st.session_state:
        st.session_state.training_completed = False

    # COMPILE AND TRAIN MODEL
    # Reset training flag if user wants to train again
    ResetTraining_col, CompileTrain_col = st.columns(2)
    
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
                # SAVE all the variables for CNN Models to run the predicitons
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
    # test_model = keras.models.load_model(model)
    # test_loss, test_acc = test_model.evaluate(test_dataset)
    
    data_eval = st.expander("Data Visualization")
    # Plotting the Model evaluation in a expander
    with data_eval:
        # st.write(f"Test accuracy: {test_acc:.3f}")
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
            st.line_chart(
                        df[["accuracy", "val_accuracy"]], 
                        y_label=["accuracy", "val_accuracy"], 
                        use_container_width=True
                        )
            
            st.line_chart(
                        df[["loss", "val_loss"]], 
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
        # Both Pytorch and Tensorflow/Keras need an order structure for the ImageClassification to recognize which images (classes) are there.
        # If the folder is named Cat, all images in this folder are used to train the CNN model on the Cat.
        # Exactly the same if the other folder is called Dog 

        train_dir = "UploadedFile/Train"
        test_dir = "UploadedFile/Test"
        validation_dir = "UploadedFile/Validation"

        train_subDir_class_1 = f"UploadedFile/Train/{name_class_1}"
        test_subDir_class_1 = f"UploadedFile/Test/{name_class_1}"
        valid_subDir_class_1 = f"UploadedFile/Validation/{name_class_1}"

        train_subDir_class_2 = f"UploadedFile/Train/{name_class_2}"
        test_subDir_class_2 = f"UploadedFile/Test/{name_class_2}"
        valid_subDir_class_2 = f"UploadedFile/Validation/{name_class_2}"
        
        # Creates a directory based on the user input on which the classes are trained.
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(validation_dir, exist_ok=True)
        
        if not os.path.exists(train_subDir_class_1):
            os.makedirs(train_subDir_class_1, exist_ok=True)
            os.makedirs(test_subDir_class_1, exist_ok=True)
            os.makedirs(valid_subDir_class_1, exist_ok=True)
        
        if not os.path.exists(train_subDir_class_2):
            os.makedirs(train_subDir_class_2, exist_ok=True)
            os.makedirs(test_subDir_class_2, exist_ok=True)
            os.makedirs(valid_subDir_class_2, exist_ok=True)

        # File upload columns
        file_uploader_col_1, file_uploader_col_2 = st.columns(2)

        # Train Dataset Upload
        with file_uploader_col_1:
            image_files_1 = st.file_uploader(
                "Upload the first Class Images", 
                accept_multiple_files=True, 
                type=["jpg", "jpeg", "png","JPEG"], 
                key="file_uploader_1")
            
            images_int = len(image_files_1)
        
            legth_dirTrain = int(images_int * .60)
            legth_dirTest = int(images_int * .20)
            legth_dirValid = int(images_int * .20)

            if image_files_1:
                # looping over the pictures to rename them in numbers (0,1,2).jepg
                for idx, image_file in enumerate(image_files_1):
                    try:
                        # Open and process image using Pillow
                        img = Image.open(image_file)
                        # checke if mode ist RGB otherwise create a error message
                        # Keras function image_dataset_from_directory need RGB otherwise error message
                        # if img.mode == "RGBA":
                        #     img = img.convert("RGB")
                        
                        # Determine save path based on index
                        if idx < legth_dirTrain:
                            save_dir = train_subDir_class_1
                        elif idx < legth_dirTrain + legth_dirTest:
                            save_dir = test_subDir_class_1
                        else:
                            save_dir = valid_subDir_class_1

                        save_path = os.path.join(save_dir, f"{os.path.basename(image_file.name)}")
                        img.save(save_path, format="PNG")
                    
                    except Exception as e:
                        st.error(f"Error processing {image_file.name}: {e}")
            

        with file_uploader_col_2:
            image_files_2 = st.file_uploader(
                "Upload the second Class Images", 
                accept_multiple_files=True, 
                type=["jpg", "jpeg", "png"], 
                    key="file_uploader_2")

            if image_files_1:
            # looping over the pictures to rename them in numbers (0,1,2).jepg
                for idx, image_file in enumerate(image_files_2):
                    try:
                        # Open and process image using Pillow
                        img = Image.open(image_file)
                        # checke if mode ist RGB otherwise create a error message
                        # Keras function image_dataset_from_directory need RGB otherwise error message
                        # if img.mode == "RGBA":
                        #     img = img.convert("RGB")
                        
                        # Determine save path based on index
                        if idx < legth_dirTrain:
                            save_dir = train_subDir_class_2
                        elif idx < legth_dirTrain + legth_dirTest:
                            save_dir = test_subDir_class_2
                        else:
                            save_dir = valid_subDir_class_2

                        save_path = os.path.join(save_dir, f"{os.path.basename(image_file.name)}")
                        img.save(save_path, format="PNG")
                    
                    except Exception as e:
                        st.error(f"Error processing {image_file.name}: {e}")
            
            
            # Prüfen, ob der Ordner existiert
            

          

            # Überprüfung der Verzeichnisse
        # Prüfen, ob das Verzeichnis existiert
        if not os.path.exists(train_subDir_class_1):
            st.error(f"The directory '{train_subDir_class_1}' does not exist. Please create it and upload the images.")
            st.stop()

        # Prüfen, ob das Verzeichnis leer ist
        if len(os.listdir(train_subDir_class_1)) == 0:
            st.warning(f"The directory '{train_subDir_class_1}' is empty. Please upload images before proceeding.")
            st.stop()

        # Prüfen, ob das Verzeichnis existiert
        if not os.path.exists(train_subDir_class_2):
            st.error(f"The directory '{train_subDir_class_2}' does not exist. Please create it and upload the images.")
            st.stop()

        # Prüfen, ob das Verzeichnis leer ist
        if len(os.listdir(train_subDir_class_2)) == 0:
            st.warning(f"The directory '{train_subDir_class_2}' is empty. Please upload images before proceeding.")
            st.stop()

        

        # as mentioed before the dir strucutre is curcial for CNN in Tensorflow.Keras
        # image_size and batch_size ist changeble
        train_dataset = image_dataset_from_directory(
            train_dir,
            image_size=(180, 180),
            batch_size=32
        )
        
        validation_dataset = image_dataset_from_directory(
            validation_dir,
            image_size=(180, 180),
            batch_size=32
        )

        test_dataset = image_dataset_from_directory(
            test_dir,
            image_size=(180, 180),
            batch_size=32
        )

    else:
        st.stop()


    st.divider()

    ##########################################
    ###### O W N _ I N F R A C T U R E #######
    ##########################################
    # Data Augmentation Expander mit Aktivierung/Deaktivierung
    data_augmentation_expander = st.expander("Data Augmentation")
    use_data_augmentation = st.checkbox("Enable Data Augmentation", value=False)

    if use_data_augmentation:
        with data_augmentation_expander:
            st.write("Configure data augmentation layers")

            augmentation_layers = st.number_input(
                "Number of Augmentation Layers", 
                min_value=1, max_value=5, step=1, 
                key="augmentation_layers"
            )

            augmentation_types = []  # Liste für die Augmentierungsschichten
            augmentation_params = []  # Liste für Parameter (falls nötig)

            for i in range(augmentation_layers):
                st.divider()
                st.subheader(f"Augmentation Layer {i+1}")
                augmentation_type = st.selectbox(
                    f'Layer {i+1} Type',
                    [
                        'RandomFlip', 'RandomRotation', 'RandomZoom',
                        'RandomTranslation', 'RandomContrast', 'RandomCrop'
                    ],
                    key=f'augmentation_type_{i}'
                )
                augmentation_types.append(augmentation_type)

                # Zusätzliche Parameter basierend auf dem Typ sammeln
                if augmentation_type == "RandomFlip":
                    flip_mode = st.selectbox(
                        f"Flip Mode (Layer {i+1})", 
                        ["horizontal", "vertical", "horizontal_and_vertical"], 
                        key=f'flip_mode_{i}'
                    )
                    augmentation_params.append({"mode": flip_mode})

                elif augmentation_type == "RandomRotation":
                    rotation_factor = st.slider(
                        f"Rotation Factor (Layer {i+1})", 
                        min_value=0.0, max_value=1.0, value=0.1, step=0.01, 
                        key=f'rotation_factor_{i}'
                    )
                    augmentation_params.append({"factor": rotation_factor})

                elif augmentation_type == "RandomZoom":
                    zoom_factor = st.slider(
                        f"Zoom Factor (Layer {i+1})", 
                        min_value=0.0, max_value=0.5, value=0.2, step=0.01, 
                        key=f'zoom_factor_{i}'
                    )
                    augmentation_params.append({"factor": zoom_factor})

                elif augmentation_type == "RandomTranslation":
                    height_factor = st.slider(
                        f"Height Factor (Layer {i+1})", 
                        min_value=0.0, max_value=0.5, value=0.1, step=0.01, 
                        key=f'height_factor_{i}'
                    )
                    width_factor = st.slider(
                        f"Width Factor (Layer {i+1})", 
                        min_value=0.0, max_value=0.5, value=0.1, step=0.01, 
                        key=f'width_factor_{i}'
                    )
                    augmentation_params.append({"height_factor": height_factor, "width_factor": width_factor})

                elif augmentation_type == "RandomContrast":
                    contrast_factor = st.slider(
                        f"Contrast Factor (Layer {i+1})", 
                        min_value=0.0, max_value=0.5, value=0.2, step=0.01, 
                        key=f'contrast_factor_{i}'
                    )
                    augmentation_params.append({"factor": contrast_factor})

                elif augmentation_type == "RandomCrop":
                    height = st.number_input(
                        f"Crop Height (Layer {i+1})", 
                        min_value=1, max_value=180, value=150, step=1, 
                        key=f'crop_height_{i}'
                    )
                    width = st.number_input(
                        f"Crop Width (Layer {i+1})", 
                        min_value=1, max_value=180, value=150, step=1, 
                        key=f'crop_width_{i}'
                    )
                    augmentation_params.append({"height": height, "width": width})

            # Dynamische Augmentierungsschichten erstellen
            st.subheader("Generated Augmentation Layers")
            augmentation_layers_list = []  # Liste der Augmentierungsschichten

            for i, aug_type in enumerate(augmentation_types):
                params = augmentation_params[i]

                if aug_type == "RandomFlip":
                    augmentation_layers_list.append(layers.RandomFlip(mode=params["mode"]))
                elif aug_type == "RandomRotation":
                    augmentation_layers_list.append(layers.RandomRotation(factor=params["factor"]))
                elif aug_type == "RandomZoom":
                    augmentation_layers_list.append(layers.RandomZoom(height_factor=params["factor"]))
                elif aug_type == "RandomTranslation":
                    augmentation_layers_list.append(layers.RandomTranslation(
                        height_factor=params["height_factor"], 
                        width_factor=params["width_factor"]
                    ))
                elif aug_type == "RandomContrast":
                    augmentation_layers_list.append(layers.RandomContrast(factor=params["factor"]))
                elif aug_type == "RandomCrop":
                    augmentation_layers_list.append(layers.RandomCrop(
                        height=params["height"], 
                        width=params["width"]
                    ))

            # Erstelle die data_augmentation Sequential-Schicht
            data_augmentation = keras.Sequential(augmentation_layers_list)
            st.write("Data Augmentation Configured:", data_augmentation)
    else:
        st.write("Data Augmentation is disabled.")
        data_augmentation = None

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
                # SAVE all the variables for CNN Models to run the predicitons
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


            
    # Ensure training is completed before testing
    if st.session_state.get('training_completed', False):
        st.divider()
        st.markdown("### Test your Model:")
        
        # Evaluate the model
        test_loss, test_acc = st.session_state.model.evaluate(test_dataset)
        st.session_state.test_acc = test_acc
        st.write(f"Test Accuracy: {test_acc:.3f}")
        
        # Check if model exists in session state
        if "model" not in st.session_state:
            st.error("Model not found. Please train the model first.")
        else:
            # File uploader for testing
            uploaded_image = st.file_uploader(
                "Upload an image to test your model",
                type=["png", "jpg", "jpeg"]
            )

            if uploaded_image is not None:
                try:
                    # Preprocess the uploaded image
                    img = image.load_img(uploaded_image, target_size=(180, 180))
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)

                    # Make predictions
                    model = st.session_state.model
                    prediction_results = model.predict(img_array)

                    # Interpret and display results
                    if prediction_results[0] > 0.5:
                        st.markdown(
                            f"The image is classified as **{name_class_1}** "
                            f"with a probability of {prediction_results[0][0]:.2f}."
                        )
                    else:
                        st.markdown(
                            f"The image is classified as **{name_class_2}** "
                            f"with a probability of {1 - prediction_results[0][0]:.2f}."
                        )
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
            else:
                st.info("Please upload an image to test the model.")
    else:
        st.error("Training not completed. Please train the model before testing.")