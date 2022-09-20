# EEL5840 - Final Project

<!-- ABOUT THE PROJECT -->
## About The Project

This repository contains the final project submission of the group **Average.Pandas** for the course EEL5840 - Fundamentals of Machine Learning. In this project, we have developed a deep learning pipeline to classify images of 10 handwritten symbols.

## Milestone 2 - Code Implementation & Technical Report

This milestone is to be delivered at the end of the semester, Friday April 22 @ 11:59 PM. Find the complete [rubric](https://ufl.instructure.com/courses/447948/assignments/5138679) in the Canvas assignemtn.

## Training Data

The training data set is the same for every team in this course.

You can download the data in our Canvas page:
* ["data_train.npy"](https://ufl.instructure.com/courses/447948/files/folder/Final%20Project?preview=67069006)
* ["labels_train.npy"](https://ufl.instructure.com/courses/447948/files/folder/Final%20Project?preview=67068769)



<!-- GETTING STARTED -->
## Getting Started

Clone the repository

  ```sh
  git clone https://github.com/EEL5840-EEE4773-Spring2022/final-project-code-and-report-average-pandas.git
  ```

### Dependencies

Here is the list of packages that need to be installed to run this project. 

* tensorflow-gpu 
  ```sh
  conda install -c anaconda tensorflow-gpu
  ```
* keras
  ```sh
  conda install -c conda-forge keras
  ```
* opencv
  ```sh
  conda install -c conda-forge opencv
  ```
* scikit-learn
  ```sh
  conda install -c anaconda scikit-learn
  ```
* matplotlib
  ```sh
  conda install -c conda-forge matplotlib
  ```
* numpy
  ```sh
  conda install -c anaconda numpy
  ```
* scipy
  ```sh
  conda install -c anaconda scipy
  ```

### Alternative: Create the Environment
Alternatively, you can create the python working environment from environment.txt file. However, since the original environment was created in a linux red hat machine, there are some dependencies on the operating system. Some of the packages might be missing in a windows machine when creating the environment.

1. Setup the environment
  ```sh
  conda create --name average_pandas --file environment.txt
  ```
2. Activate the environment
  ```sh
  conda activate average_pandas
  ```

<!-- USAGE EXAMPLES -->
## Usage

This repository contains the follwoing files: 
* ["train.py"](https://github.com/EEL5840-EEE4773-Spring2022/final-project-code-and-report-average-pandas/blob/main/train.py): training script
* ["test.py"](https://github.com/EEL5840-EEE4773-Spring2022/final-project-code-and-report-average-pandas/blob/main/test.py): testing script
* "modelCNN.hdf5": pretrained model
* "Data" directory: contains example test data and labels

The [train.py](https://github.com/EEL5840-EEE4773-Spring2022/final-project-code-and-report-average-pandas/blob/main/train.py) file is used to train the dataset with our developed deep convolutional neural network. The directory containing the training data and labels need to be modified in the variable [*file_path*](https://github.com/EEL5840-EEE4773-Spring2022/final-project-code-and-report-average-pandas/blob/bb812840d82d3cd75e61cc45bb1fe23755be9024/train.py#L45). The trained model is saved in *modelCNN.hdf5* file.

The model can be tested using the [test.py](https://github.com/EEL5840-EEE4773-Spring2022/final-project-code-and-report-average-pandas/blob/main/test.py) file. Similar to the training phase, the directory containing the test data and labels need to be modified in the variable [*file_path*](https://github.com/EEL5840-EEE4773-Spring2022/final-project-code-and-report-average-pandas/blob/c5eef15c6f927e7b680cae11f0eaf329097e4636/test.py#L43). We provided a sample test dataset in [my_test_images.npy](https://github.com/EEL5840-EEE4773-Spring2022/final-project-code-and-report-average-pandas/blob/main/Data/my_test_images.npy) and [my_test_labels.npy](https://github.com/EEL5840-EEE4773-Spring2022/final-project-code-and-report-average-pandas/blob/bb812840d82d3cd75e61cc45bb1fe23755be9024/Data/my_test_labels.npy) files for testing our model. Running the [test.py](https://github.com/EEL5840-EEE4773-Spring2022/final-project-code-and-report-average-pandas/blob/main/test.py) script will provide the accuracy, predicted and true labels as output.


<!-- Authors -->
## Authors

Your Name - Hasan Al-Shaikh (hasanalshaikh@ufl.edu), Shuvagata Saha (sh.saha@ufl.edu), Md Kawser Bepary (mdkawser.bepary@ufl.edu)

Project Link: [https://github.com/EEL5840-EEE4773-Spring2022/final-project-code-and-report-average-pandas](https://github.com/EEL5840-EEE4773-Spring2022/final-project-code-and-report-average-pandas)


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

We would like to thank the course instructor, Dr. [Catia Silva](https://faculty.eng.ufl.edu/catia-silva/bio/), and the teaching assistant, Haotian Yue, for their continuous support and motivation to finish the project on time. We would like to also thank our friend [Dipayan Saha](https://scholar.google.com/citations?user=SGJ0kqgAAAAJ&hl=en) for his suggestions and feedback in various stages of the project.



## Thank you!
