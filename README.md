# Mask2face

Can you virtually remove a face mask to see what a person looks like underneath? Our Machine Learning team proves it’s possible via an image inpainting-based ML solution. [Here](https://www.strv.com/blog/see-face-beneath-mask-how-we-built-ml-solution-engineering) is exactly how our engineers approached the problem — from the preconditions to the implementation, results and future improvements.

**Check out [the article](https://www.strv.com/blog/see-face-beneath-mask-how-we-built-ml-solution-engineering) for a more in-depth explanation**

![Example Result 1](images/result_1.png)
![Example Result 2](images/result_2.png)
**Examples of results** (input | expected output | actual output)

## How to run this project
Results can be replicated by following those steps:

### How to setup ENV
- If your **system does not have an Nvidia CUDA device available**, please comment `tensorflow-gpu==2.2.0` in the _environment.yml_ file.
- If you are running MacOS, change `tensorflow==2.2.0` to `tensorflow==2.0.0` in the _environment.yml_ file.
- Use [Conda ENV Manager](https://towardsdatascience.com/a-guide-to-conda-environments-bc6180fc533) to create new ENV: `conda env create -f environment.yml`
- Activate the ENV: `conda activate mask2face`

### Get Data
- Download [Labeled Faces in the Wild data](http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz) and unzip its content into _data_ folder or use `mask2face.ipynb` that will download it automatically.
- You can get better results using larger dataset or dataset with higher quality images. For example [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) contains over 200 000 high quality images. 

### Train the model
- Run Jupyter server in the ENV: `jupyter notebook`
- Follow instructions in the _mask2face_ notebook

### Known Issues
- There might be a memory leak when training the model 
- Performance on real-world data might not be optimal - follow our [tips]() to get possible results on real-world data
- **If you encounter any other issues, please, let us know; we are happy to help**

## Final Notes
If you’re considering our help, you may be interested in our other past work—like the [custom AI solution we built for Cinnamon in just four months](https://www.strv.com/blog/strv-cinnamon-building-custom-ai-solutions-in-4-months). And if you’re a fellow engineer, please feel free to reach out to us with any questions or share your work results. We’re always happy to start up a discussion. 
