# Assignment 2: Keypoint Detectors

## Overview

This project focuses on keypoint matching, image stitching using SIFT (Scale-Invariant Feature Transform) and RANSAC (Random Sample Consensus), and image classification. The project is divided into two main parts: Keypoint Detectors and Image Stitching.

### Requirements

#### Packages

- pillow
- matplotlib
- sklearn
- numpy
- opencv-python
- scipy
- scikit-image
- scikit-learn
- tqdm

#### Environment Setup

This project uses Poetry to manage its dependencies.
To install poetry, see the [official documentation](https://python-poetry.org/docs/).
To setup the environment using poetry, run the following command:
*Files need for poetry are `poetry.lock`, `poetry.toml` and `pyproject.toml`**

```bash
poetry install
```

Or you can install as a package using pip:

```bash
pip install ./dist/feature_matching-1.0.0.tar.gz
```

### Part 1: Keypoint Detectors

The files used for this part are:

- `load_and_split.py`
- `cifar10.npz`
- `feature_extraction.py`
- `evaluate_sift.py`
- `processed_cifar10_sift.npz`
- `feature_extraction_summary.md`

Before executing this section, run
*If installed as a package*

```bash
load_and_split
```

*If poetry installed env*

```bash
poetry run load_and_split
```

*As a script*

```bash
python ./feature_matching/keypoint_detectors/load_and_split.py
```

to load the images and split them into training and testing sets, it will be saved as **`cifar10.npz`** in the cwd. Then run
*If installed as a package*

```bash
feature_extraction
```

*If poetry installed env*

```bash
poetry run feature_extraction
```

*As a script*

```bash
python ./feature_matching/keypoint_detectors/feature_extraction.py
```

to extract the features from the images. A file named **`processed_cifar10_sift.npz`** will be saved in the cwd to be evaluated.
Finally, run

*If installed as a package*

```bash
evaluate_sift
```

*If poetry installed env*

```bash
poetry run evaluate_sift
```

*As a script*

```bash
python ./feature_matching/keypoint_detectors/evaluate_sift.py
```

to evaluate the SIFT algorithm. A report of my testing can be found in **`feature_extraction_summary.md`** of my testing results.

### Part 2: Image Stitching

The files used for this part are:

- `stitch_images.py`
- `feature_extraction.py`
- `RANSAC_Report.md`
- `data/Rainier1.png`
- `data/Rainier2.png`

**Note, functions from feature_extraction.py are used in stitch_images.py**
Run

*If installed as a package*

```bash
stitch_images
```

*If poetry installed env*

```bash
poetry run stitch_images
```

*As a script*

```bash
python ./feature_matching/keypoint_detectors/stitch_images.py
```

to stitch the images together. A report of my testings can be found in **`RANSAC_Report.md`**.
