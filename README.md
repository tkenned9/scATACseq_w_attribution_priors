# Improving Single Cell ATAC-seq Profile Models using Attribution Priors

CS273B Project - Tom Kennedy

project.py: Contains model definition and helper functions

Training.ipynb: Loads data, defines functions to train models, trains models with and without prior

Testing.ipynb: Runs best models on test set and prints all results

DeepShap.ipynb: Runs Deepshap on trained models and saves results

Interpret.ipynb: Runs Statistical tests on attribution scores from Deepshap, Creates visualizations

dinuc_shuffle.py and viz_sequence.py: helper functions lifted directly from https://github.com/amtseng/fourier_attribution_priors

data/ : Data (omitted in this repo)

runs/ : Tensorboard visualizations

shap/ : Saved attribution scores

trained_models/ : Trained models. Best ones were and exp5_epoch_17.pt (no prior) and exp6_prior_epoch_20.pt (prior). Both were run with same hyperparameters despite different experiment number in the filename.







