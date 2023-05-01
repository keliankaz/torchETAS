# Flexible and Scalable Earthquake Forecasting
This repository includes the pytorch temporal ETAS implementation. 

## Experimenting with the model
To experiment with the model please refer tp the [notebook](declustering.ipynb) in the repository.

## Installation
The code has been tested on Linux (Ubuntu 20.04) and MacOS.
1. Make sure you have the latest version of [`conda`](https://docs.conda.io/en/latest/miniconda.html) installed.
2. Install the dependencies and create a new conda environment.
    ```bash
    cd torchETAS
    conda env create -f environment.yml
    conda activate eq
    ```
   If you don't have a GPU on your machine, remove the line 
    ```
      - cudatoolkit=11.3
    ```
    from the file `environment.yml` before executing the commands above.
3. Install the `eq` package.
    ```bash
    pip install -e .
    ```

## Available models
- `eq.models.ETAS`: The classic Epidemic Type Aftershock Sequence (ETAS) model.

## Available eartquake catalogs
The code includes the following earthquake catalogs for Southern California that we used in our experiments.
| Catalog name                                                                                         | Catalog start | Catalog end | # events | Magnitude of completness |
| ---------------------------------------------------------------------------------------------------- | ------------- | ----------- | -------- | ------------------------ |
| [`eq.catalogs.White`](https://data.mendeley.com/datasets/7ywkdx7c62/1)                               | 2008-01       | 2021-01     | 134975   | 0.6                      |
| [`eq.catalogs.SCEDC`](https://service.scedc.caltech.edu/ftp/catalogs/SCEC_DC/)                       | 1981-01       | 2020-01     | 125421   | 2.0                      |
| [`eq.catalogs.QTMSaltonSea`](https://service.scedc.caltech.edu/ftp/QTMcatalog/qtm_final_12dev.hypo)  | 2008-01       | 2018-01     | 44133    | 1.0                      |
| [`eq.catalogs.QTMSanJacinto`](https://service.scedc.caltech.edu/ftp/QTMcatalog/qtm_final_12dev.hypo) | 2008-01       | 2018-01     | 20790    | 1.0                      |
In addition, the following synthetic catalogs were used in the experiments
- `eq.catalogs.ETAS_SingleCatalog`: One long catalog produced by the ETAS model.
- `eq.catalogs.ETAS_MultiCatalog`: Multiple short catalogs produces by the ETAS model.