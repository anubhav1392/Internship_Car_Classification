# Internship_Car_Classification
This is the tensorflow implementation for car model classification.
Training is done on images collected from google. Our data contains images of swift dezire and honda city and model aim is predict swift dezire car from images.
Model approach:
For feature extraction we used Resnet50 with top layer replaced by GlobalAveragePooling.
Images are resize into (128,128,3) dimensions.
Loss function used: Cross entropy with last layer using sigmoid function.
Model performance: Model is trained for 25 epochs but performance wasn't good enough.
Improvements: More images,more epochs with learning rates scheduling will solve problem.
