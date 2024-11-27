import streamlit as st
import numpy as np
import imageio.v2 as imageio  # Verwende imageio.v2, um das alte Verhalten beizubehalten
from PIL import Image
from keras.applications.mobilenet_v2 import preprocess_input
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import decode_predictions
import torch
from torchvision import transforms
from transformers import ViTImageProcessor, ViTForImageClassification

import requests

st.title("Webpage")

# Display image
image_file = st.file_uploader("Upload your Pictures")
st.image(image_file)


MobileNetV2_expander = st.expander("MobilenetV2")
with MobileNetV2_expander:
    # Load Model
    model = MobileNetV2(weights='imagenet')

    # Bild laden
    img = imageio.imread(image_file)

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


senet_expander = st.expander("senet")
with senet_expander:
    st.write("Hallo")
    model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=True)
    
    input_image = Image.open(image_file)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
    #print(output[0].shape)
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    #print(probabilities.shape)
    # Read the categories
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

        # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        st.write(categories[top5_catid[i]], top5_prob[i].item())






vit_expander = st.expander("ViT (Transformer)")
with vit_expander:
    
    # Pfad zum Bild
    image_path = image_file # Gib den vollständigen Pfad zum Bild an
    image = Image.open(image_path)

    # Processor und Modell laden
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    # Bild verarbeiten
    inputs = processor(images=image, return_tensors="pt")

    # Vorhersage machen
    outputs = model(**inputs)
    logits = outputs.logits

    # Die vorhergesagte Klasse bestimmen
    predicted_class_idx = logits.argmax(-1).item()

    # Vorhersage ausgeben
    st.write("Predicted class:", model.config.id2label[predicted_class_idx])
   
