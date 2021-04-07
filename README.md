# Mask2face
TODO - add title image from article

TODO - some short info about project (can be reused something from article), link to article, and images with results


## How to setup ENV
- Run `conda config --add channels conda-forge` to add repository to Conda
- Use Conda ENV Manager to create new ENV: `conda env create -f environment.yml`
- Activate the ENV: `conda activate mask2face`
- To run Jupyter server int the ENV: `jupyter notebook`

## Get Data
- Download **Labeled Faces in the Wild** data (http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz) and unzip its content into _data_ folder or use `mask2face.ipynb` that will download it automatically.
- You can get better results using larger dataset or dataset with higher quality images. For example [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) contains over 200 000 high quality images. 

### TODOs
- Update dependencies
- Header comments, method comments, same formatting
- Batch vs Instance vs None norm