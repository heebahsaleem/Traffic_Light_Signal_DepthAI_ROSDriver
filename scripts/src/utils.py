from src import config
import tensorflow as tf
import depthai as dai
from pathlib import Path
import numpy as np
import cv2
def prepare_batch_dataset(data_path, img_size, batch_size, shuffle=True):
	return tf.keras.preprocessing.image_dataset_from_directory(
		data_path,
		image_size=(img_size, img_size),
		shuffle=shuffle,
		batch_size=batch_size
	)

def callbacks():
	# build an early stopping callback and return it
	callbacks = [
		tf.keras.callbacks.EarlyStopping(
			monitor="val_loss",
			min_delta=0,
			patience=70000, #patience to 5% or 10% of the total epochs. for 1 million epochs,start with an early stopping patience of 50,000 to 100,000 epochs
			mode="auto",
		),
	]
	return callbacks

def normalize_layer(factor=1./127.5):
	# return a normalization layer
	return tf.keras.layers.Rescaling(factor, offset=-1)
def augmentation():
	# build a sequential model with augmentations
	data_aug = tf.keras.Sequential(
		[
			tf.keras.layers.RandomFlip("horizontal"),
			tf.keras.layers.RandomRotation(0.1),
			tf.keras.layers.RandomZoom(0.1),
		]
	)
	return data_aug


def create_pipeline_images():
	print("initializing pipeline....")
	pipeline = dai.Pipeline()
	classifierIN = pipeline.createXLinkIn()
	classifierIN.setStreamName("classifier_in")

	print("[INFO] initializingtraffic ligh classifier network...")
	classifierNN = pipeline.create(dai.node.NeuralNetwork)
	classifierNN.setBlobPath(
        str(Path(config.TRAFFIC_LIGHT_SIGNAL_CLASSIFFIER).resolve().absolute())
		
    )
	print(config.TRAFFIC_LIGHT_SIGNAL_CLASSIFFIER)
	classifierIN.out.link(classifierNN.input)
    # configure outputs for depthai pipeline
	classifierNNOut = pipeline.createXLinkOut()
	classifierNNOut.setStreamName("classifier_nn")
	classifierNN.out.link(classifierNNOut.input)
    # return the pipeline
	return pipeline

def create_pipeline_camera():
    print("[INFO] initializing pipeline...")
    # initialize a depthai pipeline
    pipeline = dai.Pipeline()
    # configure traffic light signal classifier model and set its input
    print("[INFO] initializing traffic light signal classifier network...")
    classifierNN = pipeline.create(dai.node.NeuralNetwork)
    classifierNN.setBlobPath(
        str(Path(config.TRAFFIC_LIGHT_SIGNAL_CLASSIFFIER).resolve().absolute())
    )
	# create and configure the color camera properties
    print("[INFO] Creating Color Camera...")
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(config.CAMERA_PREV_DIM)
    cam_rgb.setInterleaved(False)
    cam_rgb.setResolution(
        dai.ColorCameraProperties.SensorResolution.THE_1080_P
    )
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

	# create XLinkOut node for displaying frames
    cam_xout = pipeline.create(dai.node.XLinkOut)
    # set stream name as rgb
    cam_xout.setStreamName("rgb")
    # link the camera preview to XLinkOut node input
    cam_rgb.preview.link(cam_xout.input)

	# resize the camera frames to dimensions expected by neural network
    print("[INFO] Creating ImageManip node...")
    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(config.IMG_DIM)
    # link the camera preview to the manipulation node for resize
    cam_rgb.preview.link(manip.inputImage)
    # link the output of resized frame to input of neural network
    manip.out.link(classifierNN.input)

	 # configure outputs for depthai pipeline
    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("nn")
    classifierNN.out.link(xout_nn.input)
    # return the pipeline
    return pipeline

def softmax(x):
    # compute softmax values for each set of scores in x.
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    # resize the image array and modify the channel dimensions
    resized = cv2.resize(arr, shape)
    return resized.transpose(2, 0, 1)