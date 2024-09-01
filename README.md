# PE (Protein Engineering)

This repository aims to use machine learning to help aid protein engineering and exploring techniques devoloped for this.

The main focus is on data generated for exploring double mutations effect on epistatsis of a protein on some binding affinity metric, which is the aim to predict by using a CNN with both identity and aaindex descriptors describing each amino acid. To reduce dimensionality of the vast aaindex, PCA is used to retain the 5 dimensions with most variation.

The results is consistently generating a model with ca. 90% accuracy

## How to run

A env.yml has been provided for creating a conda environmnent

To execute the code base for now export 'PE' directory to your PYTHONPATH
Example:
export PYTHONPATH="$PYTHONPATH:home/user/somedirectory/PE/code"

### Code Usage

Run both data tranformation of aaindex and gb1 from */data_processing/transform_data*
Example:

python code/data_processing/transform_data/transform_aaindex_data.py 


Generate gb1 model by running */model_basics/model_gb1.py* (adjust for numbers of processor threads on the system for speed up)
Example:
python code/model_basics/model_gb1.py 

## In Progress

- Addition of ube4b strand
- Optimal model search (Via. grid search and stuff)


## Explanation of Protein Databases

### GB1 of protein G

GB1 of protein G is the IgG-binding domain of protein G.

The dataset details all the single and double mutations of between all the positions of the string sequence.

The fitness metric describes binding affinity to IgGFC.

The study that generated this data is exploring the effects of epistatis of mutations to a protein.

### ube4b

Possible Expansion

## Links

- https://enrich2.readthedocs.io/en/latest/introduction.html
- https://www.brenda-enzymes.org/enzyme.php?ecno=3.2.1.21

- https://www.cell.com/current-biology/fulltext/S0960-9822(14)01268-8
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3619334/
