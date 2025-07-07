# EffiSegNet Implementation Code

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10601024.svg)](https://doi.org/10.5281/zenodo.10601024)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/effisegnet-gastrointestinal-polyp/medical-image-segmentation-on-kvasir-seg)](https://paperswithcode.com/sota/medical-image-segmentation-on-kvasir-seg?p=effisegnet-gastrointestinal-polyp)

## Paper

[![arXiv](https://img.shields.io/badge/arXiv-2407.16298-b31b1b.svg)](https://arxiv.org/abs/2407.16298)

## Requirements

The environment.yml file contains the list of packages required to run the code. To create the environment, run the following command:

```bash
conda env create -f environment.yml
```

## Dataset

The dataset used for training, validation, and testing is the Kvasir-SEG dataset. The dataset can be downloaded from [here](https://datasets.simula.no/kvasir-seg/). 
However, the dataset and the splits are already available in this repository, for reproducibility of our results.
It contains 1000 images of the gastrointestinal tract and their corresponding ground truth polyp segmentation masks.
