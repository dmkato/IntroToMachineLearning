To evaluate a running prediction and gold file do as follows:

1. Please put your pred/gold files under current directory. The
name of the files can be optional, but defaults are gold.csv and
pred.csv

2. Use
python eval_simple.py
for default arguments

or

python eval_simple.py -p pred.csv -g gold.csv -o result.csv
for optional input/output names

Notes:
1- The gold file is ground truth; contains one label per row(sample)
2- The pred file is your prediction; contains score label
per row
3- The number of rows must be the same for pred and gold
