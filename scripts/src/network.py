from src.utils import augmentation
from src.utils import normalize_layer
import tensorflow as tf
class MobileNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the pretrained mobilenet feature extractor model
		# and set the base model layers to non-trainable
		base_model = tf.keras.applications.MobileNetV2(
			input_shape=(height, width, depth),
			include_top=False,
			weights="imagenet",)
		base_model.trainable = False
		inputs = tf.keras.Input(shape=(height, width, depth))
		x = augmentation()(inputs)
		x = normalize_layer()(x)
		x = base_model(x, training=False)
		x = tf.keras.layers.GlobalAveragePooling2D()(x)
		x = tf.keras.layers.Dropout(0.2)(x)
		outputs = tf.keras.layers.Dense(classes)(x)
		model = tf.keras.Model(inputs, outputs)
		return model