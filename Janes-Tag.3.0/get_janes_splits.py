import random
import conllu

# Write the train split. rnd < 0.8 should be true 80% of the time. At the end, store
# the list of ids for all sents in the train split and write to a file.
def write_train_splits(subcorpus, label_type):
    with open(f"Janes-Tag_split/train/janes-{subcorpus}_{label_type}_train.conllu", "w", encoding="UTF-8") as write_file:
        with open(f"Janes-Tag.3.0.CoNLL-U/janes-{subcorpus}.{label_type}.conllu", "r", encoding="UTF-8") as read_file:
            random.seed(6722)
            ids_in_train = []
            sentences = conllu.parse(read_file.read())
            for tokenlist in sentences:
                if "newdoc id" in tokenlist.metadata.keys():    # Documents should stay intact in each of the splits
                    rnd = random.random()
                    if rnd < 0.8:
                        ids_in_train.append(tokenlist.metadata["sent_id"])
                        write_file.write(tokenlist.serialize())
                        # Write subsequent sentences until you reach the next document or the end of the file.
                        for next_tokenlist in sentences[sentences.index(tokenlist) + 1:]:
                            if "newdoc id" in next_tokenlist.metadata.keys():
                                break
                            ids_in_train.append(next_tokenlist.metadata["sent_id"])
                            write_file.write(next_tokenlist.serialize())

    with open(f"Janes-Tag_split/train/janes-{subcorpus}_train_ids.txt", "w", encoding="UTF-8") as write_id_file:
        for ele in ids_in_train:
            write_id_file.write(ele + "\n")

    print(f"Janes-Tag_split/train/janes-{subcorpus}_{label_type}_train.conllu done!")


# Get list of all sents in subcorpus and return list of all documents that are not in the new train split
# These will go into dev and test splits
def get_list_notintrain(subcorpus):
    with open(f"Janes-Tag.3.0.CoNLL-U/janes-{subcorpus}.jos.conllu", "r", encoding="UTF-8") as read_file:
        list_all_in_subcorpus = []
        sentences = conllu.parse(read_file.read())
        for tokenlist in sentences:
            list_all_in_subcorpus.append(tokenlist.metadata["sent_id"])

    with open(f"Janes-Tag_split/train/janes-{subcorpus}_train_ids.txt", "r", encoding="utf-8") as read_file_train_ids:
        list_in_train = []
        for line in read_file_train_ids:
            list_in_train.append(line[:-1])

    list_notintrain = []
    for ele in list_all_in_subcorpus:
        if ele not in list_in_train:
            list_notintrain.append(ele)

    list_newdocs = []
    for ele in list_notintrain:
        for tokenlist in sentences:
            if ele in tokenlist.metadata["sent_id"] and "newdoc id" in tokenlist.metadata.keys():
                list_newdocs.append(ele)
                break

    return list_newdocs


# Get the list of documents that are not in the train split, shuffle the list randomly, and split it in two.
# The two halves are the dev and test splits
def write_janes_devtest(subcorpus):
    subcorpus_notintrain = get_list_notintrain(f"{subcorpus}")

    random.shuffle(subcorpus_notintrain)
    list_indev = subcorpus_notintrain[:len(subcorpus_notintrain)//2]
    list_intest = subcorpus_notintrain[len(subcorpus_notintrain)//2:]

    ids_indev = []
    ids_intest = []

    def write_split(subcorpus, label_type, split, docs_in_split, sents_in_split, conllu_sents, write_ids):
        with open(f"Janes-Tag_split/{split}/janes-{subcorpus}_{label_type}_{split}.conllu", "w",
                  encoding="UTF-8") as write_file:
            for doc_in_split in docs_in_split:
                for tokenlist in conllu_sents:
                    if doc_in_split in tokenlist.metadata["sent_id"]:
                        if write_ids:
                            sents_in_split.append(tokenlist.metadata["sent_id"])
                        write_file.write(tokenlist.serialize())
                        # Write subsequent sentences until you reach the next document or the end of the file.
                        for next_tokenlist in conllu_sents[conllu_sents.index(tokenlist) + 1:]:
                            if "newdoc id" in next_tokenlist.metadata.keys():
                                break
                            if write_ids:
                                sents_in_split.append(next_tokenlist.metadata["sent_id"])
                            write_file.write(next_tokenlist.serialize())
                        break

    def read_and_write_splits(subcorpus, label_type, write_ids=True):
        with open(f"Janes-Tag.3.0.CoNLL-U/janes-{subcorpus}.{label_type}.conllu", "r", encoding="utf-8") as read_file:
            all_sentences = conllu.parse(read_file.read())

        write_split(subcorpus, label_type, "dev", list_indev, ids_indev, all_sentences, write_ids)
        write_split(subcorpus, label_type, "test", list_intest, ids_intest, all_sentences, write_ids)


    read_and_write_splits(subcorpus, "jos")
    read_and_write_splits(subcorpus, "ud", write_ids=False)

    with open(f"Janes-Tag_split/dev/janes-{subcorpus}_dev_ids.txt", "w", encoding="UTF-8") as write_devids:
        for ele in ids_indev:
            write_devids.write(ele + "\n")

    with open(f"Janes-Tag_split/test/janes-{subcorpus}_test_ids.txt", "w", encoding="UTF-8") as write_testids:
        for ele in ids_intest:
            write_testids.write(ele + "\n")


    print(f"Janes-Tag.3.0.CoNLL-U/janes-{subcorpus}.jos.conllu & "
          f"Janes-Tag.3.0.CoNLL-U/janes-{subcorpus}.ud.conllu done!")



# At the end, check the length of all subcorpora, the no. of tokens and the percentage of sentences in each subcorpus.
def get_subcorpus_split_length(subcorpus):
    with open(f"Janes-Tag_split/train/janes-{subcorpus}_train_ids.txt", "r", encoding="UTF-8") as read_train:
        train_list = []
        for line in read_train:
            train_list.append(line[:-1])

    with open(f"Janes-Tag_split/dev/janes-{subcorpus}_dev_ids.txt", "r", encoding="UTF-8") as read_dev:
        dev_list = []
        for line in read_dev:
            dev_list.append(line[:-1])

    with open(f"Janes-Tag_split/test/janes-{subcorpus}_test_ids.txt", "r", encoding="UTF-8") as read_test:
        test_list = []
        for line in read_test:
            test_list.append(line[:-1])

    return len(train_list), len(dev_list), len(test_list)


def get_subcorpus_tokenno(subcorpus):
    with open(f"Janes-Tag_split/train/janes-{subcorpus}_jos_train.conllu", "r", encoding="UTF-8") as read_train:
        train_sents = conllu.parse(read_train.read())
        train_tokenno = 0
        for train_sent in train_sents:
            train_tokenno += len(train_sent)

    with open(f"Janes-Tag_split/dev/janes-{subcorpus}_jos_dev.conllu", "r", encoding="UTF-8") as read_dev:
        dev_sents = conllu.parse(read_dev.read())
        dev_tokenno = 0
        for dev_sent in dev_sents:
            dev_tokenno += len(dev_sent)

    with open(f"Janes-Tag_split/test/janes-{subcorpus}_jos_test.conllu", "r", encoding="UTF-8") as read_test:
        test_sents = conllu.parse(read_test.read())
        test_tokenno = 0
        for test_sent in test_sents:
            test_tokenno += len(test_sent)

    return train_tokenno, dev_tokenno, test_tokenno


def check_sizes(len_train, len_dev, len_test, tok_train, tok_dev, tok_test, subcorpus):
    len_total = len_train + len_dev + len_test
    tok_total = tok_train + tok_dev + tok_test
    print(f"{subcorpus}: \n sents in train:{len_train} \n dev:{len_dev} \n test:{len_test}")
    print(f"% sents in train:{(len_train / len_total) * 100} \n "
          f"dev:{(len_dev / len_total) * 100} \n test:{(len_test / len_total) * 100}")
    print(f"no. of tokens: \n train:{tok_train} \n dev:{tok_dev} \n test:{tok_test}")
    print(f"% of tok in train:{(tok_train / tok_total) * 100} \n "
          f"dev:{(tok_dev / tok_total) * 100} \n test:{(tok_test / tok_total) * 100}")


write_train_splits("rsdo", "jos")
write_train_splits("rsdo", "ud")
write_train_splits("tag", "jos")
write_train_splits("tag", "ud")

write_janes_devtest("rsdo")
write_janes_devtest("tag")

rsdo_train, rsdo_dev, rsdo_test = get_subcorpus_split_length("rsdo")
rsdo_train_tok, rsdo_dev_tok, rsdo_test_tok = get_subcorpus_tokenno("rsdo")

check_sizes(rsdo_train, rsdo_dev, rsdo_test, rsdo_train_tok,
            rsdo_dev_tok, rsdo_test_tok, "rsdo")

tag_train, tag_dev, tag_test = get_subcorpus_split_length("tag")
tag_train_tok, tag_dev_tok, tag_test_tok = get_subcorpus_tokenno("tag")

check_sizes(tag_train, tag_dev, tag_test, tag_train_tok,
            tag_dev_tok, tag_test_tok, "tag")

total_train = rsdo_train + tag_train
total_dev = rsdo_dev + tag_dev
total_test = rsdo_test + tag_test

total_train_tok = rsdo_train_tok + tag_train_tok
total_dev_tok = rsdo_dev_tok + tag_dev_tok
total_test_tok = rsdo_test_tok + tag_test_tok

check_sizes(total_train, total_dev, total_test, total_train_tok, total_dev_tok, total_test_tok, "Janes total")

print(f"Sents in Janes: {total_train + total_dev + total_test}")
print(f"Tokens in Janes: {total_train_tok + total_dev_tok + total_test_tok}")