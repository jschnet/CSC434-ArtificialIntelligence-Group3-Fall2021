# CSC434-ArtificialIntelligence-Group3-Fall2021
This repository serves as the submission for the final project to Dr. Yu's AI course 434 at SUNY Brockport.

These files working in part of the Kaggle competition found here: https://www.kaggle.com/c/sartorius-cell-instance-segmentation/overview

The data set was supplied in this kaggle competition, which can be found in the link above or by clicking the link here: https://www.kaggle.com/c/sartorius-cell-instance-segmentation/data

Since this project was successfully ran through the use of detectron2, which was originally developed by facebook's research and development. You can find the original repo for detectron2 here: https://github.com/facebookresearch/detectron2

There are additional resources for more information on this version of detectron2 that can be used for linux/macOS. These are the links for those resources: https://ai.facebook.com/blog/-detectron2-a-pytorch-based-modular-object-detection-library-/
https://detectron2.readthedocs.io/en/latest/tutorials/install.html

Due to the limitations of our team's systems, it was necessary to find a model that would function with windows OS and still be able to handle instance segmentation within reasonable time. With this in mind, we were about to find a rebuilt version of detectron2 that will work in windows OS. This model can be found here: https://github.com/augmentedstartups/detectron2

The team for augmentedstartups also offers a video to help with the installation process for their version of detectron2. The following link will help with building an environment that is capable of supporting this version of detectron2 (some modifications will be required to run this version depending desired output.)
Installation video: https://www.youtube.com/watch?v=JC4D9kfZdDI&t=1s

The code here was adapted to meet the specifications of the competition as well as work with the windows version of detectron2. These files are still under development, they act mainly as a benchmark of the last functional version of code with the best results so far. In the coming weeks, the remainder of our team hopes to improve the accuracy, currently 80%, and reduce the current loss, currently around 25%. Once these issues are improved, we will create a notebook with the code found here and post our findings/results to the kaggle competition. 
