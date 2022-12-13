# suk-split
The splitting code for Slovenski Učni Korpus

The existing splits for the SSJ-UD corpus were taken as the starting point. The ratio for train:dev:test for this corpus was 8:1:1 and this is preserved in the SUK splits. SSJ-UD includes the elexis-wsd and ssj500k-syn subcorpora. 

The code is contained in get_suk_splits.py. SUK_conllu contains the entire SUK corpus, while SUK_train, SUK_dev and SUK_test contain the split files and the sentence ids for each subcorpus. ssj-ud_ids contains the sentence ids for SSJ-UD. Statistics.txt contains the sizes and percentages for each subcorpus and split.

The python [conllu](https://pypi.org/project/conllu/) package is required to run the script.
