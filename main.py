import streamlit as st
import numpy as np
import imageio.v2 as imageio  # Verwende imageio.v2, um das alte Verhalten beizubehalten
from PIL import Image
from keras.applications.mobilenet_v2 import preprocess_input
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import decode_predictions



st.title("Webpage")

# Display image
image = st.file_uploader("Upload your Pictures")
st.image(image)

# Load Model
model = MobileNetV2(weights='imagenet')

# Bild laden
img = imageio.imread(image)

# Bild auf Größe (224, 224) skalieren
img_resized = Image.fromarray(img).resize((224, 224))

# Bild in das numpy Array einfügen
data = np.empty((1, 224, 224, 3))
data[0] = np.array(img_resized)
data = preprocess_input(data)

#Prediciton 
predictions = model.predict(data)

st.write('Shape: {}'.format(predictions.shape))
output_neuron = np.argmax(predictions[0])
print('Most active neuron: {} ({:.2f})'.format(
    output_neuron,
    100 * predictions[0][output_neuron]))

for name, desc, score in decode_predictions(predictions)[0]:
    st.write('- {} ({:.2f}%)'.format(desc, 100 * score))


