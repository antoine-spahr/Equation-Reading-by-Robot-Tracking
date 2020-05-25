# Equation-Reading-by-Robot-Tracking 🤖

## Goal
Read a simple handwritten equation (digits and operator) from the visual tracking of robot hovering the characters.

## Input
The input video is an `.avi` file displaying a filmed red arrow moving over different handwritten digits and operators.

![](data/robot-parcours-1.gif)

## Approach
Since the equation elements are steady in the video, they only need to be classified once from the first frame. The First frame is analyzed in three steps.
1. Get a binary mask of the object
2. Label objects and get their bounding box
3. Classify object type (digit or operators) and their value (the character represented)

The classification of type is made using K-means since operators are blue and digits are black (and the arrow is red). Due to the uniformity of the operators over videos, the classification of operator's value is made using a 5-NN models on the 5 first Fourier descriptors (translation, rotation and scale invariant). We have access to one example of of each operator, but rotating each of them by 1° gives slight variation on the mask and thus the Fourier descriptors. We can therefore train the 5-NN on 1800 samples. We obtain a test accuracy of 100% with a perfect separation of the classes. The digit are classified using a 4-layers MLP (784 -> 200 -> 100 -> 50 -> 9). Since the digit can be rotated on the original image, the MLP has to be rotation invariant. To do so we trained the MLP on randomly rotated MNIST images.

The second step is tracking the arrow and _read_ the equation. At each frame we detect the arrow as the largest object on the segmentation mask. We get the bounding box of the arrow and add the value to the equation if the arrow overlap one of the detected element.

## Result on training sequence

![](outputs/output.gif)
