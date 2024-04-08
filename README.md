## American Sign Language (ASL) Detection System
### Model Overview
- The ASL dataset is obtained from Kaggle at: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
- Total of 28 classes, including: A, B, C, D, E, F, G, H, I, J, K, L, M, 	N, O, P, Q, R, S, T, U, V, W, X, Y, Z,	DEL, SPACE
- 20 images was taken from each class for annotation and model training using YOLOv8s algorithm
</br>

### System Overview
- The inputs of images are taken using a webcam.
- The video frames are captured and mirrored for ASL detection and recognition.
- The results are displayed by drawing the bounding box around the detected sign language, attached with the detected ASL with its confidence score.
</br>

### Sample Output
![Sample Output](https://github.com/kahwei26/ASL-Detection/assets/93248505/2ef322ff-2f1a-4cfd-bf9c-d74125cc4e83)
</br>

### Limitation
- The accuracy of the ASL detection system can be further enhanced by increasing the size of the dataset fed into the model.

