# Visual Object Classes Challenge

## Goal

Develop supervised machine learning algorithm to predict or detect the presence of a class in an image.

## selected classes

5 class that will be used as test images:

- bikes
- motobikes
- people
- cats
- trains

## Current Implementation

- implemented 'load_voc_data()' to process the VOC2012 dataset
- handle multi-label classification:
  - binary labels (0: not present, 1: present)
  - skipping difficult examples
  - labels `["bicycle", "motorbike", "person", "cat", "train"]`
    - [1, 0, 0, 1, 0] represent that bicycle and cat are in the image
