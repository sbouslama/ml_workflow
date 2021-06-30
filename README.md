Description
================================================================================

This is an end-to-end workflow to approach any machine learning problem.
Please note that we will work in an IDE editor rather than jupyter notebooks.

Howerver, we will use Jupyter notebooks for data exploration and visualization.

Project Structure
================================================================================

The inside of the project folder should look something like the following.

![](img/project_structure.PNG)

Environement Setup
================================================================================
1- We use:
- [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
  to setup our environment,
- and python 3.7

Setup our environment:
```bash
conda --version
# Clone the repo
git clone ***
cd ml_workflow
# Create a conda environment. It is recommended to change the environment name with the project name. 
conda env create -f environment.yaml
# Activate the environment.
conda activate ml_workflow
# Update the environment if needed
conda env update ml_workflow --file environment.yaml

```

TRY IT
================================================================================

1- Define the needed parameters in the config file

- data path
- target column
- number of folds

2- Define model to train in the model dispatcher. Always start by a simple model.

3- Train your model from the terminal
```cmd
python train --fold 1 --model xgb
```

4- If you want to process the data, engineer features and manipulate your data, you can add some custom functions inside the 
function `run(fold, model)` in the `train.py` script 
