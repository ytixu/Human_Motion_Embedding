import os

OUTPUT_DIR = '../out/'

def create_dir(directory):
	if not os.path.exists(directory):
	    os.makedirs(directory)

def output_dir(directory):
	create_dir(OUTPUT_DIR+directory)