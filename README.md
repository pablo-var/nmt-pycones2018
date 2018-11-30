# Neural Machine Translation with attention. Pycones-2018. [Video](https://www.youtube.com/watch?v=4wenaXJlkQU)

In this repository we provide the material in order to apply the model explained in the talk as well as the presentation.

# Install

Here is the list of libraries you need to install to execute the code:
- python = 3.6
- keras
- matplotlib
- jupyter

All of them can be installed via `conda` (`anaconda`), e.g.
```
conda install jupyter
```

## Docker image

Alternatively, you can use a Docker image that exposes a Jupyter Notebook with all required dependencies. To build this image ensure you have both [docker](https://www.docker.com/)

```
docker pull pablovargasibarra/nmt-pycones2018
```

After the build you can start the container as

```
docker run --rm -it --ipc=host -p 8888:8888 pablovargasibarra/nmt-pycones2018
```

you access the repository path and activate the environment

```
cd home/nmt-pycones2018/
```

```
source activate pycones2018
```

you will be provided an URL through which you can connect to the Jupyter notebook

```
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
```

## Google Colab

Alternatively, you can use google-colab directly by uploading the notebook in google-colab/nmt-pycones.ipynb. Make sure to launch Python 3 with GPU support.
