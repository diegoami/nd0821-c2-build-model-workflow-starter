# Build an ML Pipeline for Short-Term Rental Prices in NYC

Creation of a model to predict the short rental prices in NYC, using MLFlow and Weight & Biases

# Table of Contents
* [Overview](#Overview)
** [Data Set](#Data-Set)
** [Completed Pipeline](#Completed-Pipeline)
** [Model Detail](#Model-Detail)
** [Github Release](#Github-Release)
* [Pipeline replication](#Pipeline-Replication)
** [Create environment](#Create-Environment)
** [Weights and Biases](#Weights-and-Biases)
** [Download and clean data](#Download-and-clean-data) 
** [EDA](#EDA)
** [Data testing](#Data-testing)
** [Data splitting](#Data-splitting)
** [Optimize hyperparameters](#Optimize-hyperparameters)
** [Select the best model](#Select-the-best-model)
** [Regression Test](#Regression-Test)
* 

## Overview

### Data Set 

The Data Set is contains prices of short rentals for apartments in NYC. We want to create a model to be able to predict rental prices
based on features, such as description, room type, location etc.

### Completed pipeline

A full pipeline to create a AI MODEL has been fine-tuned and executed, using MLFlow. Artifacts have been uploaed to Weights & Biases and can be found
in the public project [https://wandb.ai/diegoami/nyc_airbnb](https://wandb.ai/diegoami/nyc_airbnb)

The trained model which is giving the best result is [
random_forest_export:v13](https://wandb.ai/diegoami/nyc_airbnb/artifacts/model_export/random_forest_export/979d7469850b7b5a180e), 
which has been trained by the run [scarlet-surf-41](https://wandb.ai/diegoami/nyc_airbnb/runs/3b3g3q4n/overview), picking the one with the smallest Mean Absolute Error. 

As we can see from the [features importance graph](https://wandb.ai/diegoami/nyc_airbnb/runs/3b3g3q4n), the most importance features are the
ad name and room type.

### Model Detail

To train the random forest model, we used among others text features as tfidf vectors (such as the _name_) and categorical features
such as _room_type_ and _neighborhood group_.
To find the best model we use a grid search, the best hyperparameters have been saved in the _config.yaml_ file. 


### Github Release

The completed pipeline has been released on Github, version [1.0.3](https://github.com/diegoami/nd0821-c2-build-model-workflow-starter/releases/tag/1.0.3)

## Pipeline Replication 

### Create Environment
Mlflow is needed to execute the pipeline.
Make sure to have conda installed and ready, then create a new environment using the ``environment.yml``
file provided in the root of the repository and activate it:

```bash
> conda env create -f environment.yml
> conda activate nyc_airbnb_dev
```

### Weights and Biases
Let's make sure we are logged in to Weights & Biases. Get your API key from W&B by going to 
[https://wandb.ai/authorize](https://wandb.ai/authorize) and click on the + icon (copy to clipboard), 
then paste your key into this command:

```bash
> wandb login [your API key]
```

### Download and clean data

Go to the project root, and execute:

```bash
>  mlflow run . -P steps=download

```

Or alternatively

```bash
>  mlflow run https://github.com/diegoami/nd0821-c2-build-model-workflow-starter.git -v 1.0.3 -P steps=download,basic_cleaning
```

### EDA

To have a look at the data, execute the `eda` step:
```bash
> mlflow run src/eda
```

This will install Jupyter and all the dependencies for `pandas-profiling`, and open a Jupyter notebook instance. 
Open the _EDA.ipynb_ notebook to analyze the data

### Data testing

To define a "reference dataset". Tag the latest ``clean_sample.csv`` artifact on W&B as our 
reference dataset. Go with your browser to ``wandb.ai``, navigate to your `nyc_airbnb` project, then to the
artifact tab. Click on "clean_sample", then on the version with the ``latest`` tag. This is the
last one we produced in the previous step. Add a tag ``reference`` to it by clicking the "+"
in the Aliases section on the right:

![reference tag](images/wandb-tag-data-test.png "adding a reference tag")
 
Then run the pipeline and make sure the tests are executed and that they pass. Remember that you can run just this
step with:

```bash
> mlflow run . -P steps="data_check"
```

or alternatively

```bash
> mlflow run  https://github.com/diegoami/nd0821-c2-build-model-workflow-starter.git -v 1.0.3 -P steps="data_check"
```


### Data splitting

We use the component called ``train_val_test_split`` to extract and segregate the test set. 
For that, we execute the step `data_split`:

```bash
> mlflow run . -P steps="data_split"
```

or alternatively

```bash
> mlflow run https://github.com/diegoami/nd0821-c2-build-model-workflow-starter.git -v 1.0.3 -P steps="data_split"
```


### Optimize hyperparameters
We can now run the entire pipeline varying the hyperparameters of the Random Forest model. This can be
accomplished easily by exploiting the Hydra configuration system, using the multi-run feature (adding the `-m` option 
at the end of the `hydra_options` specification), and try setting the parameter `modeling.max_tfidf_features` to 10, 15
and 30, and the `modeling.random_forest.max_features` to 0.1, 0.33, 0.5, 0.75, 1.


```bash
> mlflow run . \
  -P steps=train_random_forest \
  -P hydra_options="modeling.random_forest.max_features=0.1,0.33,0.5,0.75,1 modeling.max_tfidf_features=10,15,30 -m"
```
or alternatively

```bash
> mlflow run https://github.com/diegoami/nd0821-c2-build-model-workflow-starter.git -v 1.0.3 \
  -P steps=train_random_forest \
  -P hydra_options="modeling.random_forest.max_features=0.1,0.33,0.5,0.75,1 modeling.max_tfidf_features=10,15,30 -m"
```

### Select the best model
Go to W&B and select the best performing model. We are going to consider the Mean Absolute Error as our target metric,
so we are going to choose the model with the lowest MAE.

![wandb](images/wandb_select_best.gif "wandb")


Go to the artifact section of the selected job, and select the 
`model_export` output artifact.  Add a ``prod`` tag to it to mark it as 
"production ready".

### Regression Test

After selecting the best performing model, execute a regression test: 

```bash
> mlflow run . -P steps=test_regression_model
```

```bash
> mlflow run https://github.com/diegoami/nd0821-c2-build-model-workflow-starter.git -v 1.0.3 -P steps=test_regression_model
```

### Test on second sample

Verify that sample

```bash
> mlflow run https://github.com/diegoami/nd0821-c2-build-model-workflow-starter.git -v 1.0.3 -P hydra_options="etl.sample='sample2.csv'"
```



