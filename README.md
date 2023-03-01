# suk-split
The splitting code for Slovenski Učni Korpus

The existing splits for the SSJ-UD corpus were taken as the starting point. The code requires that you have the ids of the sentences in each split of the SSJ-UD corpus prepared in three files prior to running (see the *ssj-ud_ids* directory). The ratio for train:dev:test for this corpus was 8:1:1 and this is preserved in the SUK splits. SSJ-UD includes the elexis-wsd and ssj500k-syn subcorpora. 

The files are split into train:dev:test according to document boundaries, so documents are kept intact across the splits. The order of the documents in the dev and test datasets is shuffled.

The code is contained in get_suk_splits.py. SUK_conllu contains the entire SUK corpus, while SUK_train, SUK_dev and SUK_test contain the split files and the sentence ids for each subcorpus. ssj-ud_ids contains the sentence ids for SSJ-UD. Statistics.txt contains the sizes and percentages for each subcorpus and split.

The same procedure was used to create splits for the [Janes-Tag 3.0](http://hdl.handle.net/11356/1732) training corpus. The original corpus, splits, split statistics and a slightly more polished (but still messy) python script are contained in the Janes-Tag.3.0 directory. 

The python [conllu](https://pypi.org/project/conllu/) package is required to run the script.
