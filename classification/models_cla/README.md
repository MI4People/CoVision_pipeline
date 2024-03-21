# Model selection and hyperparameter search

In this folder we save all experiments regarding **model selection** and **hyperparameter search** for the **classification model**. Each experiment is defined by its unique **model architecture** and **dataset** (see list of experiments below). We run multiple times each experiment, every time with a different configuration of **hyperparameters**. Regardless the dataset, we always train the model on a **trainset** and evaluate it on a **validation set**. From the latter we extract the **performance metric** that allows us to select the **best model** within experiments. The latter is then evaluated on unseen data to estimate the **generalization error**.

For this purposes, we use the **MLflow tool**. More precisely, we use **MLflow tracking server** to keep track and store relevant information of all the multiple runs within all the multiple experiments. When a "best model" is selected, the latter is registered in the **MLflow Model Registry**, indicating that such a model passed the **development stage**. Upon registration, the model can then be moved to **staging phase** or **deployment phase**.


## Directory structure

All experiments are run from the `notebooks_cla/running_experiments.ipynb` notebook and stored inside the `models_cla/mlruns` folder. The latter contains a separate sub-folder for each experiment, identified by a unique `experiment_id`, as well as a `models` sub-folder. Each experiment folder contains one sub-folder for each different run, storing parameters, metrics, artifacts, and metadata. The **artifacts** sub-folder is very important, it holds the run's model (if we decide to save it) in pickle format as well as the model's dependencies. For the best model, we also store evaluation metrics and other artifacts (such as confussion matrix figures) inside the artifacts sub-folder as well. 

The `models_cla/mlruns/models` path contains all registered models, together with usefull metadata about the model's dependencies and version. More precisely, this path does not contain any model per se, but a reference to a model's path, which must be saved on the artifact sub-folder of some run.

## List of past, ongoing, and future experiments

* **ResNet-18**: Simple resnet-18 architecture with a 1-layer output for binary classification.
	* **full_aconag**: This model is trained, and validated using the full trainset, and valset of the aconag test kit. 
	The model is then evaluated on the full aconag testset, as well as the full dataset of the deeblueag, and paramountag test kits. **\[DONE\]**
	* **full_aconag_deepblueag_paramountag**: This model is trained, and validated using the full trainsets, and valsets of the three aconag, deepblueag, and 		paramoutag test kits combined. The model is then evaluated on the testsets of the three kits combined.
	* **full_aconag_fewshot**: This model is pre-trained, and validated using the full trainset, and valset of the aconag test kit. We then further train the model 	implementing few-shot learning for the deepblueag and paramountag kits. Evaluation
* **ResNet-18_Self-Supervised**:
	* ...
* **ResNet-18_Self-Supervised_Contrastive-Learning**:
	* ...


