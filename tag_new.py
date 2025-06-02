import os
import argparse
import json
import re
from functools import cmp_to_key

BATCH_SIZE=32
LANGUAGE_IDENTIFICATIOR_BATCH_SIZE=8
MAX_LENGTH=512
TOKENIZER="ltg/norbert3-large"
SEGMENTATION_DEVICE="cuda:0"
CLASSIFICATION_DEVICE="cuda:0"
SEGMENTATION_MODEL="cuda:0"
CLASSIFICATION_MODEL="cuda:0"
SCRIPT_PATH=os.path.abspath(os.path.dirname(__file__))
LABEL_LIST_FILE = os.path.abspath(os.path.dirname(__file__)) + "/models/label_list.txt"
LABEL_CLASSES_FILE= SCRIPT_PATH + "/models/labels_classifier.txt"
LABEL_ORDER_FILE= SCRIPT_PATH + "/models/labels_order.json"
LABEL_ORDER = None
CLASS_TO_LABEL_BM = None
CLASS_TO_LABEL_NN = None
MAIN_TAG_LIST_BM = None
EQUAL_TAGS = None
ID2LABEL = None
BOKMAL_LABEL="B"
NYNORSK_LABEL="N"
BOKMAL_LABEL_ID=1
NYNORSK_LABEL_ID=2

NN_TO_BM ={
        "høfleg":"høflig",
        "eint":"ent",
        "<ikkje-clb>": "<ikke-clb>",
        "<ordenstal>": "<ordenstall>",
        "<romartal>" : "<romertall>",
        "bu": "be",
        "<st-verb>": "<s-verb>"
        }

def compare_label(t1,t2):
    global LABEL_ORDER
    val1=-1
    val2=-1
    key1 = t1 + " " + t2
    key2 = t2 + " " + t1
    if key1 in LABEL_ORDER:
        val1=LABEL_ORDER[key1]
    if key2 in LABEL_ORDER:
        val2=LABEL_ORDER[key2]
    if val1>val2:
        return -1
    return 1

def load_models_and_config():
    global TOKENIZER
    global SEGMENTATION_DEVICE
    global CLASSIFICATION_DEVICE
    global SEGMENTATION_MODEL
    global CLASSIFICATION_MODEL
    global MAX_LENGTH
    global CLASS_TO_LABEL_BM
    global CLASS_TO_LABEL_NN
    global MAIN_TAG_LIST_BM
    global MAIN_TAG_LIST_NN
    global EQUAL_TAGS
    global NN_TO_BM
    global ID2LABEL
    global BOKMAL_LABEL
    global NYNORSK_LABEL
    global LABEL_ORDER
    global LABEL_CLASSES_FILE
    global LABEL_LIST_FILE

    with open(LABEL_ORDER_FILE, "r") as f:
        LABEL_ORDER = json.load(f)
    CLASS_TO_LABEL_NN={}
    with open(LABEL_LIST_FILE, "r") as f:
        MAIN_TAG_LIST_NN = [i for i in f.read().split("\n") if i!=""]
    classes=[]
    with open(LABEL_CLASSES_FILE,"r") as f:
        class_list = [i for i in f.read().split("\n") if i!=""]
    for c in class_list:
        classes=set()
        for i in range(len(c)):
            if c[i]=="1":
                classes.add(MAIN_TAG_LIST_NN[i])
        CLASS_TO_LABEL_NN[c] = classes

    cmp_key = cmp_to_key(compare_label)

    CLASS_TO_LABEL_NN={c:sorted(list(CLASS_TO_LABEL_NN[c]),key=cmp_key) for c in CLASS_TO_LABEL_NN}
#    CLASS_TO_LABEL_NN={c:[EQUAL_TAGS[i] if i in EQUAL_TAGS else i for i in CLASS_TO_LABEL_NN[c] ]  for c in CLASS_TO_LABEL_NN }
    CLASS_TO_LABEL_BM={c:[NN_TO_BM[i] if i in NN_TO_BM else i for i in CLASS_TO_LABEL_NN[c]]   for c in CLASS_TO_LABEL_NN}


    MAIN_TAG_LIST_BM=[NN_TO_BM[i] if i in NN_TO_BM else i for i in MAIN_TAG_LIST_NN]

    MAIN_TAG_LIST_DICT_NN={MAIN_TAG_LIST_NN[i]:i for i in range(len(MAIN_TAG_LIST_NN))}
    MAIN_TAG_LIST_DICT_BM={MAIN_TAG_LIST_BM[i]:i for i in range(len(MAIN_TAG_LIST_BM))}

    for i in CLASS_TO_LABEL_NN:
        CLASS_TO_LABEL_NN[i]=[MAIN_TAG_LIST_DICT_NN[j] for j in CLASS_TO_LABEL_NN[i]]

    for i in CLASS_TO_LABEL_BM:
        CLASS_TO_LABEL_BM[i]=[MAIN_TAG_LIST_DICT_BM[j] for j in CLASS_TO_LABEL_BM[i]]

#    model_config=json.load(open(MODELS_DIR + "/sentence_segmentation/config.json","r"))
#    ID2LABEL=model_config["id2label"]
#    ID2LABEL={i:"bm" if ID2LABEL[i]==BOKMAL_LABEL else "nn" if ID2LABEL[i]==NYNORSK_LABEL else "" for i in ID2LABEL}

def matcher(o):
    return o.group(0)[0] + "\n\n" + o.group(0)[2]

def split_titles(txt):
    return [i.replace("\n"," ") for i in re.sub(r"[^.!\?](\n)([^a-z,æ,ø,å,\\ ])", matcher, txt).split("\n\n")]

def main():
    global BATCH_SIZE
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", dest="filename",
                    help="file to process. Output to stdout.", metavar="FILE")
    parser.add_argument("-bm", "--bokmal", dest="spraak", action="store_const",
                        const="bm", default="au",
                    help="Tag Bokmål")
    parser.add_argument("-nn", "--nynorsk", dest="spraak", action="store_const",
                        const="nn", default="au",
                    help="Tag Nynorsk")
    parser.add_argument("-au", "--au", dest="spraak", action="store_const",
                        const="au", default="au",
                    help="Identify the langauge (default)")
    parser.add_argument("-i", "--input-dir", dest="input_dir", type=str,
                        help="Directory to process each file in it. Operates "
                        "recursively. An output directory must be provided for "
                        "use with this option. The language is identified "
                        "automatically for each file if no language is set.",
                        metavar="FILE")
    parser.add_argument("-t", "--tsv", dest="output_tsv", action="store_const",
                        const=True, default=False, help="Output in tab-separated format.")
    parser.add_argument("-o", "--output-dir", dest="output_dir", type=str,
                        help="Directory to output tagging. Adds .json to each "
                        "input file name. Overwrites existing output files. "
                        "Tries to create the directory if it does not exist. "
                        "--input-dir must be provided for use with this option.",
                        metavar="FILE")

    parser.add_argument("-b","--batch-size", action="store", default="8",
                        type=str, required=False,
                        help="Batch size for the GPU processing.")

    parser.add_argument("-lb","--language-identificator-batch-size",
                        action="store", default="4",type=str, required=False,
                        help="Batch size for the GPU processing used in language "
                        "identification. This must be less than the normal batch "
                        "size since the whole input space of the model is utilized.")

    args = parser.parse_args()

    if args.batch_size:
        try:
            BATCH_SIZE=int(args.batch_size)
        except:
            pass

    if args.language_identificator_batch_size:
        try:
            LANGUAGE_IDENTIFICATIOR_BATCH_SIZE = int(args.language_identificator_batch_size)
        except:
            pass

    if args.filename:
        if os.path.isfile(args.filename):
            load_models_and_config()
            strs=split_titles(open(args.filename,"r").read().strip().replace("\r",""))
            for s in strs:
                tag(s, sys.stdout, args.spraak, args.output_tsv)
        else:
            print("The file " + args.filename + " could not be found.")
            exit(1)
    elif args.input_dir and args.output_dir:
            output_suf = ".tsv" if args.output_tsv else ".json"

            if not os.path.isdir(args.input_dir):
                print("The input directory " + args.input_dir  + " could not be found.")
                exit(1)

            os.makedirs(args.output_dir, exist_ok=True)

            load_models_and_config()

            for dir_path, _, files in os.walk(args.input_dir):
                for f in files:
                    f_name = os.path.join(dir_path, f)
                    output_f_name = os.path.join(args.output_dir, f) + output_suf

                    print("Input: " + f_name + ", Output: " + output_f_name)

                    with open(f_name, "r") as infile:
                        with open(output_f_name, "w") as outfile:
                            strs=split_titles(infile.read().strip().replace("\r",""))
                            for s in strs:
                                tag(s, outfile, args.spraak, args.output_tsv)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()

