# RANSAC Report

- [RANSAC Report](#ransac-report)
  - [Sample Images Used](#sample-images-used)
  - [Comparing Models](#comparing-models)
    - [Affine Model](#affine-model)
    - [Projective Model](#projective-model)

## Sample Images Used

Below are the sample images used for the RANSAC algorithm.

**Rainier1**
![Rainier1](/data/Rainier1.png "Rainier1.png")

**Rainier2**
![Rainier1](/data/Rainier2.png "Rainier1.png")

## Comparing Models

### Affine Model

- Accuracy was not consistent. When tested the deviation in accuracy was moderate to high.
- Below are the results of the Affine model:

**Keypoint Matching**
![affine-keypoints](./RANSAC_report_images/affine-keypoints.png "affine-keypoints.png")

**Stitching**
![affine-stitching](./RANSAC_report_images/affine-stitching.png "affine-stitching.png")

**Example of Inconsistency**
![affine-inconsistency](./RANSAC_report_images/affine-inconsistency.png "affine-inconsistency.png")

### Projective Model

- More accurate and consistent than the Affine model. Through testing I did not find much deviation in the accuracy of the Projective model.

**Keypoint Matching**
![projective-keypoints](./RANSAC_report_images/projective-keypoints.png "projective-keypoints.png")

**Stitching**
![projective-stitching](./RANSAC_report_images/projective-stitching.png "projective-stitching.png")
