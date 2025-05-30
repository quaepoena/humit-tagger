
BATCH_SIZE=32
LANGUAGE_IDENTIFICATIOR_BATCH_SIZE=8

TOKENIZER
SEGMENTATION_DEVICE="cuda:0"
CLASSIFICATION_DEVICE="cuda:0"
SEGMENTATION_MODEL="cuda:0"
CLASSIFICATION_MODEL="cuda:0"



def load_models_and_config():
    global TOKENIZER
    global SEGMENTATION_DEVICE
    global CLASSIFICATION_DEVICE
    global SEGMENTATION_MODEL
    global CLASSIFICATION_MODEL


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

