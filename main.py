import streamlit as st
import numpy as np
import imageio.v2 as imageio 
from PIL import Image
import pandas as pd
from keras.applications.mobilenet_v2 import preprocess_input
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import decode_predictions
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad
from keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras import layers, Model
from imageio import imread
from skimage.transform import resize
from keras.applications.mobilenet_v2 import preprocess_input
import torch
from keras.layers import Dense
from keras.applications.mobilenet_v2 import decode_predictions

from torchvision import transforms
from transformers import ViTImageProcessor, ViTForImageClassification
import requests

# Title of the web page
st.title("Webpage")

# Display image uploader

image_files = st.file_uploader("Upload your Pictures", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
for image_file in image_files:
    img = imageio.imread(image_file)

# Check if an image has been uploaded
if image_files is None:
    st.write("Please Upload an Image")
    st.stop()
else:
    # Display the uploaded image
    st.success("Uploaded successfully "+ str(int(len(image_files)))+" images")
    #st.image(image_files, use_container_width=True)
    
    # Model selection
    options = st.multiselect(
        "Which Models to test",
        ["MobileNetV2", "SENET", "ViT"],
    )
    


    

    ##################################################
    ######### M O B I L E _ N E T_ V 2 ###############
    ##################################################

   
    
    if "MobileNetV2" in options:
        MobileNetV2_expander = st.expander("MobilenetV2")
        
        with MobileNetV2_expander:
            Fine_tune_toggle = st.toggle("Finetune your Model", value=False)
            if Fine_tune_toggle == False:
                base_model = MobileNetV2(weights='imagenet', input_shape=(224, 224, 3))

            # Bild laden
            
                #img = imageio.imread(image_file) # image from file_uploader Function

                # Bild auf Größe (224, 224) skalieren
                img_resized = Image.fromarray(img).resize((224, 224))

                # Bild in das numpy Array einfügen
                data = np.empty((1, 224, 224, 3))
                data[0] = np.array(img_resized)
                data = preprocess_input(data)

                #Prediciton 
                predictions = base_model.predict(data)

                st.write('Shape: {}'.format(predictions.shape))
                output_neuron = np.argmax(predictions[0])
                print('Most active neuron: {} ({:.2f})'.format(
                    output_neuron,
                    100 * predictions[0][output_neuron]))

                for name, desc, score in decode_predictions(predictions)[0]:
                    st.write('- {} ({:.2f}%)'.format(desc, 100 * score))

            ##############################################################
            ####### F I N E _ T U N I N G ################################
            ##############################################################

            if Fine_tune_toggle == True:
                model = MobileNetV2(weights='imagenet')
                if len(image_files) == 0:
                    st.write("Please upload your Pictures")
                    st.stop()
                else:
                # empty Arry erstellen länge 40
                    data = np.empty((40, 224, 224, 3))

                    # looping over the Cats pictures
                    # TODO: Creating a Ordner Structure
                    for i in range(0, 20):

                        # TODO: rename pictures to nummeric int 
                        im = imread(r'cats_and_dogs_small/train/cats/{}.jpg'.format(i + 1))  #- Ordner struktur aufbauen
                        im = preprocess_input(im)
                        im = resize(im, output_shape=(224, 224))                             # Model nimmt nur bestimmte PIXEL_WITDH und PIXEL_LENGTH an
                        data[i] = im
                    
                    # looping over the Cats pictures
                    for i in range(0, 20):
                        # TODO: rename pictures to nummeric int 

                        im = imread(r'cats_and_dogs_small/train/dogs/{}.jpg'.format(i + 1))
                        im = preprocess_input(im)
                        im = resize(im, output_shape=(224, 224))                              # Model nimmt nur bestimmte PIXEL_WITDH und LENGTH an
                        data[i + 20] = im
                    
                    # Klassifikation erstellen 40 Bilder ingesamt die ersten 20 Hund die letzten 40 Katze
                    labels = np.empty(40, dtype=int)                
                    labels[:20] = 0
                    labels[20:] = 1

                    

                    # Add layer to the base model
                    st.markdown("Create your Mobilenetv2 Model")
                    

                    # The first 15 images for male and female cats will be used for training
                    training_data = np.empty((30, 224, 224, 3))
                    training_data[:15] = data[:15]
                    training_data[15:] = data[20:35]
                    training_labels = np.empty(30)
                    training_labels[:15] = 0
                    training_labels[15:] = 1

                    # The last 5 images for male and female cats will be used for validation
                    validation_data = np.empty((10, 224, 224, 3))
                    validation_data[:5] = data[15:20]
                    validation_data[5:] = data[35:]
                    validation_labels = np.empty(10)
                    validation_labels[:5] = 0
                    validation_labels[5:] = 1

##############################################################
####### C R E A T E _ M O D E L ##############################
##############################################################
                    
                    new_layers = []
                    for layer in model.layers[:-3]:  # Entferne alle Layer ab der drittletzten Schicht
                        new_layers.append(layer)

                    # Füge deine eigenen Layer hinzu
                    from tensorflow.keras.layers import Conv2D, Dense, Flatten

                    # Hier wird ein neuer Dense-Layer hinzugefügt
                    x = Dense(256, activation='relu')(model.layers[-5].output)
                    x = Conv2D(32, (3, 3), activation='relu', padding='same')(model.layers[-3].output)
                    x = Flatten()(x)
                    x = Dense(64, activation='relu')(model.layers[-2].output)  # z. B. der drittletzte Layer
                    x = Dense(2, activation='softmax')(x)  # Dein eigener Ausgabeschicht

                    # Erstelle das neue Modell
                    new_model = Model(inputs=model.input, outputs=x)
                    

                    # Optional: Alle Layer außer dem letzten Layer nicht trainierbar machen
                    for layer in new_model.layers[:-1]:
                        layer.trainable = False
                    


                    new_model.compile(
                        optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
                    
                    new_model.fit(
                        x=training_data,
                        y=training_labels,
                        validation_data=(validation_data, validation_labels),
                        epochs=5,
                        verbose=2)
                    # Bild auf Größe (224, 224) skalieren
                    
                    for image_file in image_files:
                        if image_file is not None:
                            # Open the image using PIL
                            img = Image.open(image_file)

                            # Resize the image to (224, 224)
                            img_resized = img.resize((224, 224))

                            # Convert the image to a NumPy array
                            img_array = np.array(img_resized)

                            # Ensure the array has the correct shape for the model input (1, 224, 224, 3)
                            data = np.empty((1, 224, 224, 3))
                            data[0] = img_array

                            # Make a prediction using the model
                            predictions = new_model.predict(data)

                            predictions_df = pd.DataFrame({
                                'Label': ['Cat', 'Dog'],
                                'Probability': predictions[0]
                            })

                            # Zeige den DataFrame in der Streamlit-App an
                            st.markdown("## Predictions")
                            st.dataframe(predictions_df)



































































    #############################################
    ######### S E N E T #########################
    #############################################
# if "SENET" in options:
#     senet_expander = st.expander("SENET")
#     with senet_expander:
        
#         model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=True)
        
#         input_image = Image.open(image_file)
#         preprocess = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])
#         input_tensor = preprocess(input_image)
#         input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

#         # move the input and model to GPU for speed if available
#         if torch.cuda.is_available():
#             input_batch = input_batch.to('cuda')
#             model.to('cuda')

#         with torch.no_grad():
#             output = model(input_batch)
#         # Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
#         #print(output[0].shape)
#         # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
#         probabilities = torch.nn.functional.softmax(output[0], dim=0)
#         #print(probabilities.shape)
#         # Read the categories
#         with open("imagenet_classes.txt", "r") as f:
#             categories = [s.strip() for s in f.readlines()]

#             # Show top categories per image
#         top5_prob, top5_catid = torch.topk(probabilities, 5)
#         for i in range(top5_prob.size(0)):
#             st.write(categories[top5_catid[i]], top5_prob[i].item())






    # vit_expander = st.expander("ViT (Transformer)")
    # with vit_expander:
        
    #     # Pfad zum Bild
    #     image_path = image_file # Gib den vollständigen Pfad zum Bild an
    #     image = Image.open(image_path)

    #     # Processor und Modell laden
    #     processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    #     model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    #     # Bild verarbeiten
    #     inputs = processor(images=image, return_tensors="pt")

    #     # Vorhersage machen
    #     outputs = model(**inputs)
    #     logits = outputs.logits

    #     # Die vorhergesagte Klasse bestimmen
    #     predicted_class_idx = logits.argmax(-1).item()

    #     # Vorhersage ausgeben
    #     st.write("Predicted class:", model.config.id2label[predicted_class_idx])
