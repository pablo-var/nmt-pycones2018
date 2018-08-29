# Neural Machine Translation with attention. Pycones-2018

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

## Docker image (in process)

Alternatively, you can use a Docker image that exposes a Jupyter Notebook with all required dependencies. To build this image ensure you have both [docker](https://www.docker.com/)

```
docker build nmt-pycones18 .
```

After the build you can start the container as

```
docker run --rm -it --ipc=host -p 8888:8888 nmt-pycones18
```

you will be provided an URL through which you can connect to the Jupyter notebook.
