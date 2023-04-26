# SignLanguageImageRecognition
A sign language image classification using transfer learning from MobileNet model. 

A deep CNN implementation that modifies the last few layers of the pretrained MobileNet model and repurposes it to classify sign language digit images.

Depending on the selected subset of training and testing images, 99 to 100% accuracy is usually achieved.

The code needs tensorflow, keras, numpy, pandas, matplotlib, sklearn, itertools and glob installed in the Python environment.

Download the dataset from: https://github.com/ardamavi/Sign-Language-Digits-Dataset
![image](https://user-images.githubusercontent.com/40482921/234691854-28418994-2e7f-46bb-9b3a-2a863047401c.png)

Extract the contents of train folder to "./Sign-Language-Digits-Dataset/" folder under the project folder

During the first run the code will do a random sampling of the data and form the train, test, validation sets. They will be reused in the subsequent runs. Number of images in each set can be modified as needed. If a new train, validation & test set is needed, delete the "test", "train", and "valid" folders under "./Sign-Language-Digits-Dataset/"

The code will save the trained model and use it the next time it is called. To retrain a model, go to "./models/" folder under the project library and delete the "sign_language_model.h5" file.

GPU parallelization is turned off, but it can be turned on by uncommenting the relevant line.
