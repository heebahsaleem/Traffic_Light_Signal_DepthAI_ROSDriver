# import the necessary packages
import os
import glob
# define the base path, paths to separate train, validation, and test splits
BASE_DATASET_PATH = "/home/sonnet/Downloads/all-techpol-GN-2930"
# TRAIN_DATA_PATH = os.path.join(BASE_DATASET_PATH, "train")
# VALID_DATA_PATH = os.path.join(BASE_DATASET_PATH, "validation")
# TEST_DATA_PATH = os.path.join(BASE_DATASET_PATH, "test")

TRAIN_DATA_PATH = "/home/sonnet/Downloads/all-techpol-GN-2930/train"
VALID_DATA_PATH = "/home/sonnet/Downloads/all-techpol-GN-2930/eval"
TEST_DATA_PATH = "/home/sonnet/Downloads/all-techpol-GN-2930/eval"
OUTPUT_PATH = "output"
PICKLE_OUTPUT_PATH = "output/pickle_files"
EACH_EPOCH = "output/epochs"
TENSORBOARD_LOG_DIR = "output/tensorboard_logs"

# define the image size and the batch size of the dataset
IMAGE_SIZE = 224
BATCH_SIZE = 32 # --> 1. try 64
# number of channels, 1 for gray scale and 3 for color images
CHANNELS = 3
# define the classifier network learning rate
LR_INIT = 0.0001
# number of epochs for training
NUM_EPOCHS = 1400
# number of categories/classes in the dataset
N_CLASSES = 6
# define paths to store training plots, testing prediction and trained model
ACCURACY_LOSS_PLOT_PATH = os.path.join("/home/sonnet/Downloads/all-techpol-GN-2930/output", "accuracy_loss_plot.png")
TRAINED_MODEL_PATH = os.path.join("/home/sonnet/Downloads/all-techpol-GN-2930/output", "Traffic_Light_classifier")
TEST_PREDICTION_OUTPUT = os.path.join("/home/sonnet/Downloads/all-techpol-GN-2930/output", "test_prediction_images.png")

# below code is after conversion of model from .pd --> .oonx --> .xml & .bin --> .blob 
#MODEL_PATH ="/home/sonnet/Downloads/Traffic_Signal_Sonnet/IR _model_Correct"
MODEL_PATH ="/home/sonnet/Downloads"
#MODEL_BLOB_PATH = "model1.blob"
TRAFFIC_LIGHT_SIGNAL_CLASSIFFIER = os.path.join(MODEL_PATH,"modelVI.blob")

TEST_DATA = glob.glob("test_data/*.jpg")
OUTPUT_IMAGES = os.path.join("results","pred_images")
OUTPUT_VIDEOS = os.path.join("results","pred_camera.mov")

IMG_DIM = (300, 300)
CAMERA_PREV_DIM = (480,480)

LABELS = [
    "green" , "green-left" , "left" , "yellow", "red" , "off" ,
]