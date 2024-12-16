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
st.divider()
options = st.selectbox(
    "Which Models to test",
    ["Own Model","MobileNetV2", 
        "SENET", "ViT"],
)

if "Own Model" in options:
    st.write("Own Model")
    # User input for the number of layers
    number_layers = st.number_input("Number of Layers", min_value=1, max_value=3, step=1)

    layer_types = []
    units = []
    activations = []

    for i in range(number_layers):
        st.write(f"Layer {i+1}")

        layer_col, units_col, activation_col = st.columns(3)


        with layer_col:
            layer_type = st.selectbox(f'Layer {i+1} Type', 
                                        ['Dense', 'Conv2D', 'Maxpooling', 'Flatten'], 
                                        key=f'layer_type_{i}')
            layer_types.append(layer_type)


        with units_col:
            unit_int = st.number_input(f'Units in Layer {i+1}', 
                                    min_value=1, max_value=512, 
                                    value=64, step=1, key=f'units_{i}')
            units.append(unit_int)


        with activation_col:
            activation = st.selectbox(f'Activation in Layer {i+1}', 
                                        ['relu', 'sigmoid', 'tanh', 'softmax'], 
                                        key=f'activation_{i}')
            activations.append(activation)


    # Create model thorw user Input
    st.write("Model Created")


    inputs = keras.Input(shape=(180, 180, 3))
    data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
    ])
    
    for i in range(number_layers):
        x = data_augmentation(inputs)
    
        if layer_types[i] == "Dense":
            x = layers.Dense(units[i], activation=activations[i])(x)
        elif layer_types[i] == "Conv2D":
            x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
        elif layer_types[i] == "Maxpooling":
            x = layers.MaxPooling2D(pool_size=2)(x)
        elif layer_types[i] == "Flatten":
            x = layers.Flatten()(x)
        
    #st.write(x.summary())
    

    inputs = keras.Input(shape=(180, 180, 3))
    # x = data_augmentation(inputs)
    # x = layers.Rescaling(1./255)(x)
    # x = layers.layer_types(filters=32, kernel_size=3, activation="relu")(x)
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



    

