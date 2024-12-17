import streamlit as st
import imageio.v2 as imageio 
from tensorflow import keras
from tensorflow.keras import layers

# Title
st.title("STATY AI")

# Display image uploader



file_uploder_col_1, file_uploder_col_2 = st.columns(2)
with file_uploder_col_1:
    image_files_1 = st.file_uploader("Upload your Pictures", accept_multiple_files=True, type=["jpg", "jpeg", "png"], key="file_uploader_1")
    for image_file in image_files_1:
        img = imageio.imread(image_file)

    # Check if an image has been uploaded
    if image_files_1 is None:
        st.write("Please Upload an Image")
        st.stop()
    else:
        # Display the uploaded image
        st.success("Uploaded successfully "+ str(int(len(image_files_1)))+" images")
        #st.image(image_files, use_container_width=True)


with file_uploder_col_2:
    image_files_2 = st.file_uploader("Upload your Pictures", accept_multiple_files=True, type=["jpg", "jpeg", "png"], key="file_uploader_2")

    for image_file in image_files_2:
        img = imageio.imread(image_file)

    # Check if an image has been uploaded
    if image_files_2 is None:
        st.write("Please Upload an Image")
        st.stop()
    else:
        # Display the uploaded image
        st.success("Uploaded successfully "+ str(int(len(image_files_2)))+" images")
        #st.image(image_files, use_container_width=True)
    
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
            ['Dense', 'Conv2D', 'Maxpooling', 'Flatten'], 
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
    x = data_augmentation(inputs)  # Apply data augmentation once

    # Build layers dynamically
    for i in range(number_layers):
        layer_type = layer_types[i]
        
        if layer_type == "Conv2D":
            x = layers.Conv2D(filters=filters[i], kernel_size=kernels[i], activation=activations[i])(x)
        
        elif layer_type == "Dense":
            # Add Dense layer only after Flatten
            if "Flatten" not in layer_types[:i]:
                st.warning(f"Dense layer at position {i+1} might need a preceding Flatten layer.")
            x = layers.Dense(units[i], activation=activations[i])(x)

        elif layer_type == "MaxPooling2D":
            x = layers.MaxPooling2D(pool_size=pool_sizes[i])(x)

        elif layer_type == "Flatten":
            x = layers.Flatten()(x)

    # Add output layer
    
    st.write("Specify Output Layer")
    output_units = st.number_input("Output Units", min_value=1, value=1, step=1)
    
    output_activation = st.selectbox("Output Activation", ["sigmoid", "softmax", "linear"])
    
    outputs = layers.Dense(output_units, activation=output_activation)(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)





    # inputs = keras.Input(shape=(180, 180, 3))
    # x = data_augmentation(inputs)
    # x = layers.Rescaling(1./255)(x)
    # x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    # x = layers.MaxPooling2D(pool_size=2)(x)
    # x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    # x = layers.MaxPooling2D(pool_size=2)(x)
    # x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    # x = layers.MaxPooling2D(pool_size=2)(x)
    # x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    # x = layers.MaxPooling2D(pool_size=2)(x)
    # x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    # x = layers.Flatten()(x)
    # x = layers.Dropout(0.5)(x)
    # outputs = layers.Dense(1, activation="sigmoid")(x)
    # model = keras.Model(inputs=inputs, outputs=outputs)


    # st.write("Model Summary")
    # st.write("Model Training")
    # st.write("Model Evaluation")
    # st.write("Model Prediction")
    # st.write("Model Visualization")



    

