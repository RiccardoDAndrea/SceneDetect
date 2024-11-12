import os, shutil

original_dataset_dir = 'Data'

base_dir = r'C:\Users\riandrea\Desktop\Github\SceneDetect\cats_and_dogs_small'


if os.path.exists(base_dir) == False:
    os.mkdir(base_dir)

    train_dir = os.path.join(base_dir, 'train')
    os.mkdir(train_dir)

    validation_dir = os.path.join(base_dir, 'validation')
    os.mkdir(validation_dir)

    test_dir = os.path.join(base_dir, 'test')
    os.mkdir(test_dir)

else:
    print("Direcotry already existens")