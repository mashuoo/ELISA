## ELISA
This is a python program to draw elisa 4-parameter logistic standard curve，ELISA samples absorbance barplot and estimating ELISA samples concentration.
## Dependency
* numpy==1.21.5
* pandas==1.4.2
* scikit_learn==1.0.2
* scipy==1.7.3
* sympy==1.10.1
* matplotlib==3.5.1
## Usage
The script elisa.py is used to draw elisa 4-parameter logistic standard curve，ELISA samples absorbance barplot and estimating ELISA samples concentration.
The required arguments
* STD_DATA: ELISA standard absorbance data as given example format
* ELISA_DATA: ELISA samples absorbance data as given example format
* DILUTION_RATIO: dilution ratio of ELISA samples absorbance
* OUT_DIR: output file 
This script ouput the results for given data.

```python elisa.py -s Elisa_s.xlsx -e Elisa.xlsx -d d_r.txt -od out_dir```
