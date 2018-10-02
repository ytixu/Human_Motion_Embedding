import os
import numpy as np

# For ouputs

OUTPUT_DIR = '../out/'

def create_dir(directory):
	if not os.path.exists(directory):
	    os.makedirs(directory)

def output_dir(directory):
	create_dir(OUTPUT_DIR+directory)

# normalizations

def wrap_angle(rad, center=0):
	return ( rad - center + np.pi) % (2 * np.pi ) - np.pi

