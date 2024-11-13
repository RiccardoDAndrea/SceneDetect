import os, shutil

#original_dataset_dir_cat = 'Data\Cat'

#original_dataset_dir_dog = 'Data\Dog'

def create_dir(original_dataset_dir_cat = str, original_dataset_dir_dog=str):
    """
    Desc:

    Input:

    Output:
    
    """
    
    
    base_dir = r'C:\Users\riandrea\Desktop\Github\SceneDetect\cats_and_dogs_small'
    train_dir = None
    test_dir = None
    validation_dir = None

    if os.path.exists(base_dir) == False:
        os.mkdir(base_dir)

        train_dir = os.path.join(base_dir, 'train')
        os.mkdir(train_dir)

        validation_dir = os.path.join(base_dir, 'validation')
        os.mkdir(validation_dir)

        test_dir = os.path.join(base_dir, 'test')
        os.mkdir(test_dir)

        test_cats_dir = os.path.join(test_dir, 'cats')
        os.mkdir(test_cats_dir)

        test_dogs_dir = os.path.join(test_dir, 'dogs')
        os.mkdir(test_dogs_dir)

        train_cats_dir = os.path.join(train_dir, 'cats')
        os.mkdir(train_cats_dir)

        train_dogs_dir = os.path.join(train_dir, 'dogs')
        os.mkdir(train_dogs_dir)

        validation_cats_dir = os.path.join(validation_dir, 'cats')
        os.mkdir(validation_cats_dir)

        validation_dogs_dir = os.path.join(validation_dir, 'dogs')
        os.mkdir(validation_dogs_dir)
        
        fnames = ['{}.jpg'.format(i) for i in range(1000)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir_cat, fname)
            dst = os.path.join(train_cats_dir, fname)
            shutil.copyfile(src, dst)
        
        fnames = ['{}.jpg'.format(i) for i in range(1000, 1500)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir_cat, fname)
            dst = os.path.join(validation_cats_dir, fname)
            shutil.copyfile(src, dst)
        
        fnames = ['{}.jpg'.format(i) for i in range(1500, 2000)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir_cat, fname)
            dst = os.path.join(test_cats_dir, fname)
            shutil.copyfile(src, dst)
        
        fnames = ['{}.jpg'.format(i) for i in range(1000)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir_dog, fname)
            dst = os.path.join(train_dogs_dir, fname)
            shutil.copyfile(src, dst)
        
        fnames = ['{}.jpg'.format(i) for i in range(1000, 1500)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir_dog, fname)
            dst = os.path.join(validation_dogs_dir, fname)
            shutil.copyfile(src, dst)
        
        fnames = ['{}.jpg'.format(i) for i in range(1500, 2000)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir_dog, fname)
            dst = os.path.join(test_dogs_dir, fname)
            shutil.copyfile(src, dst)

        print("Succesfully created Dir")


    else:
        print('total test cat images:', 
            len(os.listdir(base_dir + "/test/cats/")))
        
        print('total test cat images:', 
            len(os.listdir(base_dir + "/test/dogs/")))
        
        print('total training cat images:', 
            len(os.listdir(base_dir + "/train/cats/")))
        
        print('total training cat images:', 
            len(os.listdir(base_dir + "/train/dogs/")))
        
        print('total validation cat images:', 
            len(os.listdir(base_dir + "/validation/cats/")))
        
        print('total validation cat images:', 
            len(os.listdir(base_dir + "/validation/dogs/")))
        
        train_dir = 'cats_and_dogs_small/train'
        validation_dir = 'cats_and_dogs_small/validation'
    
    return train_dir, test_dir, validation_dir

train_dir, test_dir, validation_dir = create_dir(original_dataset_dir_cat='Data\Cat', original_dataset_dir_dog='Data\Dog')

print(train_dir, test_dir, validation_dir)



from keras import layers, models, optimizers

def model_Conv():
    model = models.Sequential()
    model.add(layers.Conv2D(32,(3,3), 
                            activation='relu',
                            input_shape=(150,150,3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128,(3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128,(3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))


    model.compile(loss='binary_crossentropy',
                        optimizer=optimizers.RMSprop(learning_rate=1e-4),
                        metrics=['acc'])
    print(model.summary())
    return None

# model_Conv()

from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary')

print(train_generator)