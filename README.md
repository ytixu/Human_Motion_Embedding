# Human Motion Embedding

### Dependencies

-Tensorflow
-Keras

### Quickstart

1. Clone this repo and download the dataset in exponential map

'''
git clone https://github.com/ytixu/Human_Motion_Embedding.git
cd Human_Motion_Embedding
mkdir data
cd data
wget http://www.cs.stanford.edu/people/ashesh/h3.6m.zip
unzip h3.6m.zip
rm h3.6m.zip
cd ..
'''

2. Preprocess data
- Downsample from 50fps to 25 fps
- Convert to euler angle or cartesian coordinates, or keep in exponential map
'''
cd src/utils
python data_preprocessing.py
'''
Default conversion to euler angle. See input options for other parameterizations.