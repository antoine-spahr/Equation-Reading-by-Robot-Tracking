# Equation-Reading-by-Robot-Tracking 🤖

## Goal
Read a simple handwritten equation (digits and operator) from the visual tracking of robot hovering the characters.

## Input
The input video is an `.avi` file displaying a filmed red arrow moving over different handwritten digits and operators.

<video width="420" height="240" controls>
  <source src="/data/robot_parcours_1.avi" type="video/avi">
</video>

## Approach
Since the equation elements are steady in the video, they only need to be classified once from the first frame. The First frame is analyzed in three steps.
1. Get a binary mask of the object
2. Label objects and get their bounding box
3. Classify object type (digit or operators) and their value (the character represented)

The classification of type is made using K-means since operators are blue and digits are black (and the arrow is red). Due to the uniformity of the operators over videos, the classification of operator's value is made using a 5-NN models on the 5 first Fourier descriptors (translation, rotation and scale invariant). 
