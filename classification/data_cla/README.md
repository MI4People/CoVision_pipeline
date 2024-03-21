# README

This folder contains all pre-processed data for training, and evaluating the classification model. The data is (and must be) organized following specific conventions: each kit has its own folder, whose name is `kit_id`, in lowercase (e.g. `aconag`). Each of these folders contains a `kit_id_membranes` folder and a `kit_id_labels.csv` file. The former contains membrane images in `.jpg` format. The csv file contains image filenames as rows and the corresponding label/ground truth as columns, as well as an extra column for specifiying the data mode (namely, train, val, or test). 

The labels are in string format, containing a binary value for each line in the membrane (1 for line, 0 for no line). For two-line tests, they are one of the following options: "11", "10", "01", "00".

