# IHS 2021 Code Repository

The official code repository corresponding to the results obtained in [COVID-19 Severity Prediction from Lung Radiography Images Using Deep Learning](./IHS_2021_Submission.pdf).

## Relevant Directories

The relevant directory structure of the project is organized as such:

```
root
|-- metadata
|   `-- metadata.csv
...
|-- demos
|   |-- Test_Slices.ipynb
|   `-- Generate_Metadata.ipynb
|-- utils
|   ...
`-- multiclass_VGG16_basic.py
```

`metadata/metadata.csv` is the file responsible for specifying the files to be used in the Datasets used in `multiclass_VGG16_basic.py`, which is responsible for performing the main analysis described in the paper. Metadata file creation can be done by modifying the `Generate_Metadata.ipynb` notebook present in the `demos` folder.

Do note that the actual IEEE COVID-19 CT data used are not provided in this repository and should be found using the links provided in the paper.

## Running the Code

Running the code should simply be a matter of running `multiclass_VGG16_basic.py`. The necessary libraries are listed below.

```
numpy==1.19.2
pandas==1.1.3
scikit-learn==0.24.1
tensorboard==2.5.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.0
torch==1.8.0+cu111
torchvision==0.9.0+cu111
```

Even though the results listed are subpar, we believe our work holds value in suggesting possible avenues by which researchers may avoid the challenges we faced.

*- Zitian, Danni, Peter, Sofia, and Lawrence*
