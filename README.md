# P-unit model

Cell specific integrate-and-fire type models for a population of
P-unit electroreceptors of Apteronotus leptorhynchus

## How to simulate the models

The file `models.csv` contains the model parameters for each cell in a
table.

The file `models_202106.csv` contains model parameters from a newer
fitting attempt.

The `load_models()` function in `model.py` can be used to load these
 parameters into a list of dictionaries.

You may use the `simulate()` function in `model.py` to simulate a
model for a given stimulus.

In `main.py` you can see how to do this.

The EOD of a single fish always has an amplitude of one. The EOD
frequency is in the table with the model parameters.
