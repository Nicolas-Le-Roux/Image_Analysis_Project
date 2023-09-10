# Image Analysis Project - SOCMINTEX
This is a compilation of the work of Nicolas Le Roux, graduate student of ENSTA Bretagne, Brest, France, during his internship at the Royal Military Academy, Brussels, Belgium. Accomplished between May and September, 2023, the work consisted in exploring image analysis possibilities for a future addition to the pre-existing SOCMINTEX project.

## SOCMINTEX Background
The SOCMINTEX project deals with analysis of social media posts and interactions, with the aim to develop an ergonomic toolbox for monitoring and analyzing social media content. The front-end, as well as the back-end related to analysis of messages, was designed by Dr. Juan Carlos Fernandez Toledano. His guidance also shaped the code written for the image analysis component.

## Purpose of the code
The code contains two classes in a .py file. These classes are meant to simplify the use of computer vision models. Although Detectron2 has made working with such models much easier, the process can still be streamlined for the specific application of the SOCMINTEX project.

## Installation and other parameters
As new releases come out, the versions used during the project months may no longer be the most recent version. As such, only problematic modules, ones that caused problems when trying to install, have their versions listed.
- **Language and version :** Python 3.9
- **Environment used :** Conda
- **Detectron2 :** Before downloading the other modules, the following instructions were entered into the cluster console.
  - conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia
  - conda install -c conda-forge pycocotools
  - pip install opencv-python
  - python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
- **Other modules:**
  - copy
  - glob
  - itertools
  - json
  - matplotlib
  - numpy
  - pandas
  - re
  - time

**Remark**: The PyTorch version used is a nightly build, meaning it is not a full release. This was a necessary choice since the working CUDA version was too advanced for PyTorch's full releases at the time. See at https://pytorch.org/get-started/locally/ for the available versions.
