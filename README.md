# Image Analysis Project - SOCMINTEX
This is a compilation of the work of Nicolas Le Roux, graduate student of ENSTA Bretagne, Brest, France, during his internship at the Royal Military Academy, Brussels, Belgium. Accomplished between May and September, 2023, the work consisted in developping an image analysis component to the pre-existing SOCMINTEX project, which until then had been limited to text analysis.

## SOCMINTEX Background
The SOCMINTEX project dealt with analysis of social media posts and interactions, with the aim to develop an ergonomic toolbox for monitoring and analyzing social media content. The front-end, as well as the back-end related to analysis of messages, was designed by Dr. Juan Carlos Fernandez Toledano, whose advice was instrumental in adapting the image analysis code to the pre-existing structure.

## Purpose of the code
The code contains a series of classes, all contained within one superclass. These classes are meant to simplify the use of computer vision models, although Detectron2 has made working with such models much easier, the process can still be streamlined for the specific application of the SOCMINTEX project. According to the SOLID principles, it is designed to be added to by future users for their specific tasks.

## Installation and other parameters
As new releases come out, these may no longer be the most recent versions of the modules required to run the code. The following is the working configuration used in summer 2023, during the course of the internship.
- Language and version : Python 3.9
- Environment used : Conda
- Modules :
  - ...
  - ...

**Remark**: The PyTorch version used is a nightly build, meaning it is not a full release. This was a necessary choice since the working CUDA version was too advanced for PyTorch's full releases at the time.
