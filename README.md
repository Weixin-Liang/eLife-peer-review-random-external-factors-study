# How random is the review outcome? A systematic study of the impact of external factors on eLife peer review

This is the code and data used to conduct the analysis for the paper titled "How random is the review outcome? A systematic study of the impact of external factors on eLife peer review". 

## Abstract
*The advance of science rests on a robust peer review process. However whether or not a paper is accepted can depend on random external factors--e.g. the timing of the submission, the assignment of editors and reviewers--that are beyond the quality of the work. This article systematically investigates the impact of such random factors independent of the paper’s quality on peer review outcomes in a major biomedical journal, eLife. We analyzed all of the submissions to eLife between 2012 to 2018, with 34,161 total submissions. We examined how random factors affect each decision step of the review process from the gate-keeping senior editors who may desk-reject papers to review editors and reviewers who recommend the final outcome. Our results showed that the peer-review process in eLife is robust overall and that random external factors have relatively little quantifiable bias.*



## Usage
### 0. Dependencies

- [Python](<https://www.python.org/>) == 3.7
- [PyTorch](<https://pytorch.org/get-started/locally/>) == 1.4.0
- [transformers](<https://github.com/huggingface/transformers/tree/v2.0.0>) == 2.0.0
- [torch-geometric](https://pytorch-geometric.readthedocs.io/) ==1.6.0

Run the following commands to install the dependencies:
```bash

pip install pandas matplotlib tqdm
pip install scipy nltk bs4 
pip install fuzzywuzzy
<!-- conda install -c conda-forge fuzzywuzzy -->
pip install ethnicolr
<!-- conda install -c soodoku ethnicolr -->
```
