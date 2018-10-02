# Human Motion Embedding

### Dependencies

- [Tensorflow](https://www.tensorflow.org/) 1.2
- [Keras](https://keras.io/) 2.1
- [Tqdm](https://github.com/noamraph/tqdm) (progress bar)

### Quickstart

1. Clone this repo and download the dataset in exponential map

```bash
git clone https://github.com/ytixu/Human_Motion_Embedding.git
cd Human_Motion_Embedding
mkdir data
cd data
wget http://www.cs.stanford.edu/people/ashesh/h3.6m.zip
unzip h3.6m.zip
rm h3.6m.zip
cd ..
```

2. Preprocess data
- Downsample from 50fps to 25 fps
- Convert to euler angle or cartesian coordinates, or keep in exponential map
```bash
cd src/utils
python data_preprocessing.py
```
Default conversion to euler angle. See input options for other parameterizations and for visualization of the motions.

3. Test `parser.py`
```bash
python parser.py -m test -id ../../data/h3.6m/euler
```