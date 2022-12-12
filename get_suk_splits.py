import conllu
import random

random.seed(257)


# Read sent ids for ssj-ud splits from prepared files
def read_ids(split):
    with open(f"./ssj-ud_ids/sl_ssj-ud-{split}_ids.txt", "r", encoding="UTF-8") as read_file:
        id_list = []
        for line in read_file:
            id_list.append(line[:-1])
        return id_list


ssjud_train_ids = read_ids("train")
ssjud_dev_ids = read_ids("dev")
ssjud_test_ids = read_ids("test")


# Find ids in elexiswsd and ssj500k parts of SUK and write to new file
def write_ssjud_splits(split, subcorpus, id_list, label_type):
    with open(f"SUK_{split}/SUK_{subcorpus}_{label_type}_{split}.conllu", "w", encoding="UTF-8") as write_file:
        with open(f"SUK_conllu/{subcorpus}.{label_type}.conllu", "r", encoding="UTF-8") as read_file:
            sentences = conllu.parse(read_file.read())
            for prepared_id in id_list:
                for tokenlist in sentences:
                    if prepared_id in tokenlist.metadata["sent_id"]:
                        write_file.write(tokenlist.serialize())
                        break
    print(f"SUK_{split}/SUK_{subcorpus}_{label_type}_{split}.conllu done!")


write_ssjud_splits("train", "ssj500k-syn", ssjud_train_ids, "jos")
write_ssjud_splits("dev", "ssj500k-syn", ssjud_dev_ids, "jos")
write_ssjud_splits("test", "ssj500k-syn", ssjud_test_ids, "jos")
write_ssjud_splits("train", "ssj500k-syn", ssjud_train_ids, "ud")
write_ssjud_splits("dev", "ssj500k-syn", ssjud_dev_ids, "ud")
write_ssjud_splits("test", "ssj500k-syn", ssjud_test_ids, "ud")
write_ssjud_splits("train", "elexiswsd", ssjud_train_ids, "jos")
write_ssjud_splits("dev", "elexiswsd", ssjud_dev_ids, "jos")
write_ssjud_splits("test", "elexiswsd", ssjud_test_ids, "jos")
write_ssjud_splits("train", "elexiswsd", ssjud_train_ids, "ud")
write_ssjud_splits("dev", "elexiswsd", ssjud_dev_ids, "ud")
write_ssjud_splits("test", "elexiswsd", ssjud_test_ids, "ud")


# Write the train split for the other subcorpora. rnd < 0.87 should be true 87% of the time. A value higher than 0.8
# is needed to ensure around 80% of all sentences are included, not only documents. At the end, store the list of ids
# for all sents in train split and write to a file.
def write_suk_train_splits(subcorpus, label_type):
    with open(f"SUK_train/SUK_{subcorpus}_{label_type}_train.conllu", "w", encoding="UTF-8") as write_file:
        with open(f"SUK_conllu/{subcorpus}.{label_type}.conllu", "r", encoding="UTF-8") as read_file:
            ids_in_train = []
            sentences = conllu.parse(read_file.read())
            for tokenlist in sentences:
                if "newdoc id" in tokenlist.metadata.keys():    # Documents should stay intact in each of the splits
                    rnd = random.random()
                    if rnd < 0.87:
                        ids_in_train.append(tokenlist.metadata["sent_id"])
                        write_file.write(tokenlist.serialize())
                        # Write subsequent sentences until you reach the next document or the end of the file.
                        for next_tokenlist in sentences[sentences.index(tokenlist) + 1:]:
                            if "newdoc id" in next_tokenlist.metadata.keys():
                                break
                            ids_in_train.append(next_tokenlist.metadata["sent_id"])
                            write_file.write(next_tokenlist.serialize())

    with open(f"SUK_train/SUK_{subcorpus}_train_ids.txt", "w", encoding="UTF-8") as write_id_file:
        for ele in ids_in_train:
            write_id_file.write(ele + "\n")

    print(f"SUK_train/SUK_{subcorpus}_{label_type}_train.conllu done!")


write_suk_train_splits("senticoref", "jos")
write_suk_train_splits("senticoref", "ud")
write_suk_train_splits("ssj500k-tag", "jos")
write_suk_train_splits("ssj500k-tag", "ud")
write_suk_train_splits("ambiga", "jos")
write_suk_train_splits("ambiga", "ud")


# Get list of all sents in subcorpus and return list of those that are not in the new train split
# These will go into dev and test splits
def get_list_notintrain(subcorpus):
    with open(f"SUK_conllu/{subcorpus}.jos.conllu", "r", encoding="UTF-8") as read_file:
        list_all_in_subcorpus = []
        sentences = conllu.parse(read_file.read())
        for tokenlist in sentences:
            list_all_in_subcorpus.append(tokenlist.metadata["sent_id"])

    with open(f"SUK_train/SUK_{subcorpus}_train_ids.txt") as read_file_train_ids:
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


# Get the list of sentences that are not in the train split, shuffle the list randomly, and split it in two.
# The two halves are the dev and test splits
def write_suk_devtest(subcorpus):
    subcorpus_notintrain = get_list_notintrain(f"{subcorpus}")

    random.shuffle(subcorpus_notintrain)
    list_indev = subcorpus_notintrain[:len(subcorpus_notintrain)//2]
    list_intest = subcorpus_notintrain[len(subcorpus_notintrain)//2:]

    ids_indev = []
    ids_intest = []
    with open(f"SUK_conllu/{subcorpus}.jos.conllu", "r", encoding="UTF-8") as read_file:
        all_sentences = conllu.parse(read_file.read())
        with open(f"SUK_dev/SUK_{subcorpus}_jos_dev.conllu", "w", encoding="UTF-8") as write_dev:
            for sent_indev in list_indev:
                for tokenlist in all_sentences:
                    if sent_indev in tokenlist.metadata["sent_id"]:
                        ids_indev.append(tokenlist.metadata["sent_id"])
                        write_dev.write(tokenlist.serialize())
                        # Write subsequent sentences until you reach the next document or the end of the file.
                        for next_tokenlist in all_sentences[all_sentences.index(tokenlist) + 1:]:
                            if "newdoc id" in next_tokenlist.metadata.keys():
                                break
                            ids_indev.append(next_tokenlist.metadata["sent_id"])
                            write_dev.write(next_tokenlist.serialize())
                        break

        with open(f"SUK_test/SUK_{subcorpus}_jos_test.conllu", "w", encoding="UTF-8") as write_test:
            for sent_intest in list_intest:
                for tokenlist in all_sentences:
                    if sent_intest in tokenlist.metadata["sent_id"]:
                        ids_intest.append(tokenlist.metadata["sent_id"])
                        write_test.write(tokenlist.serialize())
                        # Write subsequent sentences until you reach the next document or the end of the file.
                        for next_tokenlist in all_sentences[all_sentences.index(tokenlist) + 1:]:
                            if "newdoc id" in next_tokenlist.metadata.keys():
                                break
                            ids_intest.append(next_tokenlist.metadata["sent_id"])
                            write_test.write(next_tokenlist.serialize())
                        break

    with open(f"SUK_conllu/{subcorpus}.ud.conllu", "r", encoding="UTF-8") as read_file:
        all_sentences = conllu.parse(read_file.read())
        with open(f"SUK_dev/SUK_{subcorpus}_ud_dev.conllu", "w", encoding="UTF-8") as write_dev:
            for sent_indev in list_indev:
                for tokenlist in all_sentences:
                    if sent_indev in tokenlist.metadata["sent_id"]:
                        ids_indev.append(tokenlist.metadata["sent_id"])
                        write_dev.write(tokenlist.serialize())
                        # Write subsequent sentences until you reach the next document or the end of the file.
                        for next_tokenlist in all_sentences[all_sentences.index(tokenlist) + 1:]:
                            if "newdoc id" in next_tokenlist.metadata.keys():
                                break
                            ids_indev.append(next_tokenlist.metadata["sent_id"])
                            write_dev.write(next_tokenlist.serialize())
                        break

        with open(f"SUK_test/SUK_{subcorpus}_ud_test.conllu", "w", encoding="UTF-8") as write_test:
            for sent_intest in list_intest:
                for tokenlist in all_sentences:
                    if sent_intest in tokenlist.metadata["sent_id"]:
                        ids_intest.append(tokenlist.metadata["sent_id"])
                        write_test.write(tokenlist.serialize())
                        # Write subsequent sentences until you reach the next document or the end of the file.
                        for next_tokenlist in all_sentences[all_sentences.index(tokenlist) + 1:]:
                            if "newdoc id" in next_tokenlist.metadata.keys():
                                break
                            ids_intest.append(next_tokenlist.metadata["sent_id"])
                            write_test.write(next_tokenlist.serialize())
                        break

    with open(f"SUK_dev/SUK_{subcorpus}_dev_ids.txt", "w", encoding="UTF-8") as write_devids:
        for ele in ids_indev:
            write_devids.write(ele + "\n")

    with open(f"SUK_test/SUK_{subcorpus}_test_ids.txt", "w", encoding="UTF-8") as write_testids:
        for ele in ids_intest:
            write_testids.write(ele + "\n")

    print(f"SUK_conllu/{subcorpus}.jos.conllu & SUK_conllu/{subcorpus}.ud.conllu done!")


write_suk_devtest("senticoref")
write_suk_devtest("ambiga")
write_suk_devtest("ssj500k-tag")


# At the end, check the length of all subcorpora and the percentage of sentences in each subcorpus.
def get_subcorpus_split_length(subcorpus):
    with open(f"SUK_train/SUK_{subcorpus}_train_ids.txt", "r", encoding="UTF-8") as read_train:
        train_list = []
        for line in read_train:
            train_list.append(line[:-1])

    with open(f"SUK_dev/SUK_{subcorpus}_dev_ids.txt", "r", encoding="UTF-8") as read_dev:
        dev_list = []
        for line in read_dev:
            dev_list.append(line[:-1])

    with open(f"SUK_test/SUK_{subcorpus}_test_ids.txt", "r", encoding="UTF-8") as read_test:
        test_list = []
        for line in read_test:
            test_list.append(line[:-1])

    return len(train_list), len(dev_list), len(test_list)


ssjud_total = len(ssjud_train_ids) + len(ssjud_dev_ids) + len(ssjud_test_ids)
print(f"ssj-ud: \n train:{len(ssjud_train_ids)} \n dev:{len(ssjud_dev_ids)} \n test:{len(ssjud_test_ids)}")
print(f"% in train:{(len(ssjud_train_ids)/ssjud_total)*100} \n dev:{(len(ssjud_dev_ids)/ssjud_total)*100} \n "
      f"test:{(len(ssjud_test_ids)/ssjud_total)*100}")

ambiga_train, ambiga_dev, ambiga_test = get_subcorpus_split_length("ambiga")
ambiga_total = ambiga_train + ambiga_dev + ambiga_test
print(f"ambiga: \n train:{ambiga_train} \n dev:{ambiga_dev} \n test:{ambiga_test}")
print(f"% in train:{(ambiga_train/ambiga_total) * 100} \n dev:{(ambiga_dev/ambiga_total)*100} \n "
      f"test:{(ambiga_test/ambiga_total)*100}")

senticoref_train, senticoref_dev, senticoref_test = get_subcorpus_split_length("senticoref")
senticoref_total = senticoref_train + senticoref_dev + senticoref_test
print(f"senticoref: \n train:{senticoref_train} \n dev:{senticoref_dev} \n test:{senticoref_test}")
print(f"% in train:{(senticoref_train/senticoref_total) * 100} \n dev:{(senticoref_dev/senticoref_total)*100} \n "
      f"test:{(senticoref_test/senticoref_total)*100}")

ssj500k_tag_train, ssj500k_tag_dev, ssj500k_tag_test = get_subcorpus_split_length("ssj500k-tag")
ssj500k_tag_total = ssj500k_tag_train + ssj500k_tag_dev + ssj500k_tag_test
print(f"ssj500k-tag: \n train:{ssj500k_tag_train} \n dev:{ssj500k_tag_dev} \n test:{ssj500k_tag_test}")
print(f"% in train:{(ssj500k_tag_train/ssj500k_tag_total) * 100} \n dev:{(ssj500k_tag_dev/ssj500k_tag_total)*100} \n "
      f"test:{(ssj500k_tag_test/ssj500k_tag_total)*100}")

total_train = len(ssjud_train_ids) + ssj500k_tag_train + ambiga_train + senticoref_train
total_dev = len(ssjud_dev_ids) + ssj500k_tag_dev + ambiga_dev + senticoref_dev
total_test = len(ssjud_test_ids) + ssj500k_tag_test + ambiga_test + senticoref_test
suk_total = total_train + total_dev + total_test
print(f"SUK total: \n train:{total_train} \n dev:{total_dev} \n test:{total_test}")
print(f"% in train:{(total_train/suk_total) * 100} \n dev:{(total_dev/suk_total)*100} \n "
      f"test:{(total_test/suk_total)*100}")
