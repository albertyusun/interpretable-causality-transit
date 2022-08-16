# Making the World More Equal, One Ride at a Time: Studying Public Transportation Initiatives Using Interpretable Causal Inference

This folder contains code for Duke Data Science Team's entry for [Stanford CISIL Causal Inference Challenge](https://casbs.stanford.edu/causal-inference-social-impact-lab-s-data-challenge).

## Authors

Duke University Data Science Team:

- Albert Sun*
- Jenny Huang*
- Gaurav Rajesh Parikh*
- Lesia Semenova
- Cynthia Rudin

Note: * Denotes equal contribution

## Folder Organization

1. FLAME_SAP_Reenrollment.py - This file includes the code we used to run our analyses for understanding the effect of SAP on _reenrollment_.

2. FLAME_SAP_Ridership.py - This file includes the code we used to run our analyses for understanding the effect of SAP on _ridership_.

3. requirements.txt - This file includes the Python packages we used for our analyses.

## Packages

The Python packages we used are in `requirements.txt`. Notably, we used the following packages _a lot_ to run our analyses.

- dame-flame,
- numpy,
- pandas,
- scikit-learn,

## Reproducibility

Most of the data that we used in our code are open source, namely the US Census [American Community Survey Data](https://www.census.gov/programs-surveys/acs/data.html) and Urban Institute's [Unequal Commute data](https://www.urban.org/features/unequal-commute).

We also used ORCA-LIFT Smart Card data to conduct specific individual-level analysis. To receive this data, request it from the Stanford Causal Inference for Social Impact lab or the King County Metro.
