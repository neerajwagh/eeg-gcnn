# EEG-GCNN: Augmenting Electroencephalogram-based Neurological Disease Diagnosis using a Domain-guided Graph Convolutional Neural Network

_*Authors*: Neeraj Wagh, Yogatheesan Varatharajah_

_*Affiliation*: Department of Bioengineering, University of Illinois at Urbana-Champaign_

## Work accepted in proceedings of the ML4H Workshop, NeurIPS 2020 with an oral spotlight!

- ArXiv Pre-print: <https://arxiv.org/abs/2011.12107>
- PMLR Paper: <http://proceedings.mlr.press/v136/wagh20a.html>
- ML4H Poster: <https://drive.google.com/file/d/14nuAQKiIud3p6-c8r9WLV2tAvCyRwRev/view?usp=sharing>
- ML4H 10-minute Video: <https://slideslive.com/38941020/eeggcnn-augmenting-electroencephalogrambased-neurological-disease-diagnosis-using-a-domainguided-graph-convolutional-neural-network?ref=account-folder-62123-folders>
- ML4H Slides: <https://drive.google.com/file/d/1dXT4QAUXKauf7CAkhrVyhR2PFUsNh4b8/view?usp=sharing>
- Code: [GitHub Repo](https://github.com/neerajwagh/eeg-gcnn)
- Final Models, Pre-computed Features, Training Metadata: [FigShare .zip](https://figshare.com/articles/software/EEG-GCNN_Supporting_Resources_for_Reproducibility/13251452)
- Raw Data: [MPI LEMON](http://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html) (no registration needed), [TUH EEG Abnormal Corpus](https://www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_abnormal/) ([needs registration](https://www.isip.piconepress.com/projects/tuh_eeg/html/request_access.php))
- DGL Implementation of EEG-GCNN: [DGL EEG-GCNN page](https://github.com/dmlc/dgl/tree/master/examples/pytorch/eeg-gcnn)
### Example code to process your own EEG datasets and generate features for EEG-GCNN model (or any other model) training/evaluation: 1) prepare_data_for_eeg-gcnn.ipynb and 2) eeg_pipeline.py

### Follow these steps to execute model comparison experiments and reproduce reported results

1. Download 1) the pre-computed feature arrays for all 10-second windows in the dataset (power spectral density features, geodesic electrode distances, spectral coherence values), 2) final models used in Table 2 comparisons, and 3) training metadata (which window maps to which subject, target labels, sample indices, etc.) from [FigShare](https://figshare.com/articles/software/EEG-GCNN_Supporting_Resources_for_Reproducibility/13251452)
2. Place all the feature files and relevant model files inside the directory of the experiment you want to execute. The code expects these files to be present in the experiment's root folder.
3. Ensure your execution environment has the following Python dependencies installed and working:
    - Python 3.x
    - PyTorch (at least 1.4.0 for PyTorch Geometric to work)
    - PyTorch Geometric
    - Scikit-learn
4. Enter the directory of the experiment you want to run.
5. Execute `$ python heldout_test_run.py` to run the saved 10 final models on the held-out 30% test set of subjects using the pre-computed features. For trivial classifiers, run `$ python chance_level_classification.py`. The mean and standard deviation values reported in Table 2 of the paper will be printed at the end of execution. See notes below for more details.

### Notes

- This repository contains code to load final models and reproduce held-out test set results reported in Table 2 of the ML4H paper. All the code required to run an experiment is contained entirely inside the corresponding experiment folder.
- All experiments were run and results were reported for a fixed seed (42) in the entire pipeline. We have not repeated the experiments multiple times using different seeds due to time constraints. The seed determines 1) which subjects (out of the total available in the pooled dataset) are held-out for final testing, and 2) which subjects form the 10 train/validation folds within 10-fold cross-validation. Therefore, to reproduce reported results _exactly_, you will need to use seed as 42 since this will ensure evaluation is done on the subjects that were not seen during training of the released final models.
- We encourage you to 1) closely inspect the process flow depicted in Figure 4 to fully understand the evaluation setup, and 2) train new models using different seeds.
- The EEG-GCNN model definition can be found in EEGGraphConvNet.py in the shallow/deep EEG-GCNN experiment folders.
- The trivial classifiers are not trained on data, and only rely on the label/class imabalance information to provide chance-based predictions. To switch between the two trivial models reported in the paper, change the "MODEL_TYPE" variable in the script.

### Quirks in the Codebase

- In the code, subject-level predictions and metrics (as opposed to window-level) are held in the "_patient_" variables. For purposes of the code, "patient" = "subject", irrespective of whether the subject/s are healthy or diseased.
- For the FCNN experiment, the model is contained in the EEGConvNet.py file (although it is not a convolutional network). Changing the class/file name would make the saved models unusable and hence has been left unchanged.
- Raw channels are converted to an 8-channel bipolar montage before being used for modeling. While these 8 channels were derived in the 10-20 system, the spatial connectivity between these montage channels is calculated between the electrodes in the center of the montage channel pair in the 10-10 system. The idealized locations of the scalp electrodes in the 10-10 configuration system are taken from standard_1010.tsv file, and used in get_sensor_distances() in EEGGraphDataset.py.

### Update
After removing type cast error in edge weights calculation, the performance stays similar to what is published in the paper.
|                  | AUC          | Precision    | Recall       | F-1          | Bal. Accuracy |
|------------------|--------------|--------------|--------------|--------------|---------------|
| Shallow EEG-GCNN | 0.871(0.001) | 0.989(0.003) | 0.677(0.018) | 0.804(0.011) | 0.810(0.003)  |
| Deep EEG-GCNN    | 0.863(0.001) | 0.988(0.006) | 0.660(0.031) | 0.791(0.020) | 0.800(0.002)  |

### Contact

- Issues regarding non-reproducibility of results or support with the codebase should be emailed to Neeraj and John.
- Neeraj: nwagh2@illinois.edu / [Website](http://neerajwagh.com/) / [Twitter](https://twitter.com/neeraj_wagh) / [Google Scholar](https://scholar.google.com/citations?hl=en&user=lCy5VsUAAAAJ)
- John: wei33@illinois.edu
- Yoga: varatha2@illinois.edu / [Website](https://sites.google.com/view/yoga-personal/home) / [Google Scholar](https://scholar.google.com/citations?user=XwL4dBgAAAAJ&hl=en)

### Citation

Wagh, N. & Varatharajah, Y.. (2020). EEG-GCNN: Augmenting Electroencephalogram-based Neurological Disease Diagnosis using a Domain-guided Graph Convolutional Neural Network. Proceedings of the Machine Learning for Health NeurIPS Workshop, in PMLR 136:367-378 Available from http://proceedings.mlr.press/v136/wagh20a.html.
