# gene_expression_prediction

environmental installation guide: https://modality-docs.biomodal.com/installation.html

TL;DR Installation guide:
1. create conda env with python=3.11
2. activate conda env, install latest modality release:
  ``` pip install --extra-index-url https://europe-python.pkg.dev/prj-biomodal-modality/modality-pypi/simple modality ```
3. install `hstlib` and ``
  ```conda install -c bioconda htslib samtools```