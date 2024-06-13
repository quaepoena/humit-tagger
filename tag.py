import argparse
import os
import json
import sys
import torch
import transformers
import copy
from transformers import BertTokenizerFast
from transformers import EncoderDecoderModel
from transformers import BertModel
from transformers import BertConfig
from transformers import BertLMHeadModel
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from pynvml import *
from functools import cmp_to_key

import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

os.environ["TOKENIZERS_PARALLELISM"]="false"
enc_max_length=512

# Process os.environ["CUDA_VISIBLE_DEVICES"]
CUDA_TO_NV={}
splitted_device_list=[]

MODEL_SENTENCE_START_ID=101
MODEL_SENTENCE_END_ID=102
BATCH_SIZE=8
if "CUDA_VISIBLE_DEVICES" in os.environ:
    splitted_device_list=os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    num = 0
    for i in splitted_device_list:
        try:
            dev_id=int(i)
            CUDA_TO_NV[num]=dev_id
            num+=1
        except:
            pass
else:
    CUDA_TO_NV=None

int_classification_device=-1  # -1 for cpu or gpu id
int_tokenization_device=-1    # -1 for cpu or gpu id 
int_segmentation_device=-1  # -1 for cpu or gpu id

three_model_devices=[]
two_model_devices=[]
one_model_devices=[]

not_used_devices=[]

final_devices=[]

gpu_scores=[]

one_model_size=8000000000

nvmlInit()

def get_one_gpu(final_devices, one_model_devices, two_model_devices, three_model_devices):
    # Find one more available GPU:
    found_one=False
    for dev_num in final_devices:
        if dev_num in three_model_devices or dev_num in two_model_devices:
            final_devices.append(dev_num)
            found_one=True
            break
    if not found_one:
        for dev_num in three_model_devices:
            if dev_num not in final_devices:
                final_devices.append(dev_num)
                found_one=True
                break
            if not found_one:
                for dev_num in two_model_devices:
                    if dev_num not in final_devices:
                        final_devices.append(dev_num)
                        found_one=True
                        break
                if not found_one:
                    for dev_num in one_model_devices:
                        if dev_num not in final_devices:
                            final_devices.append(dev_num)
                            found_one=True
                            break
    return final_devices, found_one

# Try to find three GPUs with enough memory
for dev_num in range(int(torch.cuda.device_count())):
    #t = torch.cuda.get_device_properties(dev_num).total_memory
    #r = torch.cuda.memory_reserved(dev_num)
    if CUDA_TO_NV is not None:
        h = nvmlDeviceGetHandleByIndex(CUDA_TO_NV[dev_num])
    else:
        h = nvmlDeviceGetHandleByIndex(dev_num)

    info = nvmlDeviceGetMemoryInfo(h)

    # If three models fit
    if info.free > one_model_size*3:
        three_model_devices.append(dev_num)
        if info.used< 340000000 :
            not_used_devices.append(dev_num)

    # If two models fit
    elif info.free > one_model_size*2:
        two_model_devices.append(dev_num)
        if info.used< 340000000 :
            not_used_devices.append(dev_num)

    # If one model fits
    elif info.free > one_model_size:
        one_model_devices.append(dev_num)
        if info.used< 340000000 :
            not_used_devices.append(dev_num)

if len(not_used_devices)>=3:
    final_devices=not_used_devices[0:3]

elif len(not_used_devices)==2:
    final_devices.append(not_used_devices[0])
    final_devices.append(not_used_devices[1])
    final_devices, found_one = get_one_gpu(final_devices, one_model_devices, two_model_devices, three_model_devices)

elif len(not_used_devices)==1:
    final_devices.append(not_used_devices[0])
    final_devices, found_one = get_one_gpu(final_devices, one_model_devices, two_model_devices, three_model_devices)
    final_devices, found_one = get_one_gpu(final_devices, one_model_devices, two_model_devices, three_model_devices)

if len(final_devices)==1:
    for dev_num in final_devices:
        if dev_num in three_model_devices:
            if len(final_devices)<3:
                final_devices.append(dev_num)
            if len(final_devices)<3:
                final_devices.append(dev_num)
            break
        elif dev_num in two_model_devices:
            if len(final_devices)<3:
                final_devices.append(dev_num)
            if len(final_devices)==3:
                break
elif len(final_devices)==2:
    for dev_num in final_devices:
        if dev_num in three_model_devices:
            if len(final_devices)<3:
                final_devices.append(dev_num)
            break
        elif dev_num in two_model_devices:
            if len(final_devices)<3:
                final_devices.append(dev_num)
            break



# Finally set devices
if len(final_devices)==1:
    int_classification_device=final_devices[0]
elif len(final_devices)==2:
    int_classification_device=final_devices[0]
    int_tokenization_device=final_devices[1]
elif len(final_devices)==3:
    int_classification_device=final_devices[0]
    int_tokenization_device=final_devices[1]
    int_segmentation_device=final_devices[2]

label_text_file = "./models/label_list.txt"
classes_file= "./models/labels_classifier.txt"
with open("./models/labels_order.json", "r") as f:
    label_order = json.load(f)

def compare_label(t1,t2):
    global label_order
    val1=-1
    val2=-1
    key1 = t1 + " " + t2
    key2 = t2 + " " + t1
    if key1 in label_order:
        val1=label_order[key1]
    if key2 in label_order:
        val2=label_order[key2]
    if val1>val2:
        return -1
    return 1

cmp_key = cmp_to_key(compare_label)

IGNORE_BERT_TAGS={"$punc$"}
EQUAL_TAGS={":subst:":"subst",
            ":ukjent:": "ukjent",
            ":adj:":"adj",
            ":prep:":"prep",
            ":verb:":"verb",
            ":det:":"det",
            ":konj:":"konj",
            ":pron:":"pron",
            ":adv:":"adv",
            ":inf-merke:":"inf-merke",
            ":<anf>:":"<anf>",
            ":sbu:":"sbu",
            ":clb:":"clb",
            ":<komma>:":"<komma>",
            ":<strek>:":"<strek>",
            ":<parentes-beg>:":"<parentes-beg>",
            ":<parentes-slutt>:":"<parentes-slutt>",
            ":interj:":"interj",
            ":symb:":"symb"
            }
NN_TO_BM ={
        "høfleg":"høflig",
        "eint":"ent",
        "<ikkje-clb>": "<ikke-clb>",
        "<ordenstal>": "<ordenstall>",
        "<romartal>" : "<romertall>",
        "bu": "be",
        "<st-verb>": "<s-verb>"
        }

class_to_label_nn={}
with open(label_text_file, "r") as f:
    label_list= [i for i in f.read().split("\n") if i!=""]
classes=[]
with open(classes_file,"r") as f:
    class_list = [i for i in f.read().split("\n") if i!=""]
for c in class_list:
    classes=set()
    for i in range(len(c)):
        if c[i]=="1":
            classes.add(label_list[i])
    class_to_label_nn[c] = classes


class_to_label_nn={c:sorted(list(class_to_label_nn[c] - IGNORE_BERT_TAGS),key=cmp_key) for c in class_to_label_nn}

class_to_label_nn={c:[EQUAL_TAGS[i] if i in EQUAL_TAGS else i for i in class_to_label_nn[c] ]  for c in class_to_label_nn }

class_to_label_bm={c:NN_TO_BM[c] if c in NN_TO_BM  else class_to_label_nn[c]   for c in class_to_label_nn}


if int_classification_device == -1:
    classification_device="cpu"
else:
    classification_device="cuda:" + str(int_classification_device)

if int_tokenization_device == -1:
    tokenization_device="cpu"
else:
    tokenization_device="cuda:" + str(int_tokenization_device)

if int_segmentation_device == -1:
    segmentation_device="cpu"
else:
    segmentation_device="cuda:" + str(int_segmentation_device)


tag_enc_tokenizer = BertTokenizerFast.from_pretrained('NbAiLab/nb-bert-base')
segmentation_tokenizer = tag_enc_tokenizer
segmentation_tokenizer.model_max_length=512

MAX_LENGTH_WITHOUT_CLS=segmentation_tokenizer.model_max_length-1

classification_model = AutoModelForTokenClassification.from_pretrained("./models/classification/")
classification_model.to(classification_device)
classification_model.eval()

tokenization_model = AutoModelForTokenClassification.from_pretrained("./models/tokenization/")
tokenization_model.to(tokenization_device)
tokenization_model.eval()
tokenization_pipeline = pipeline('ner', model = tokenization_model, tokenizer = tag_enc_tokenizer, device = int_tokenization_device)

segmentation_model = AutoModelForTokenClassification.from_pretrained("./models/sentence_segmentation/")
segmentation_model.to(segmentation_device)
segmentation_model.eval()

all_vocab=open("./models/sentence_segmentation/vocab.txt","r").read().replace("\r","").split("\n")
TOKENS_STARTING_WITH_HASH=torch.zeros( len(all_vocab), dtype=torch.bool, device=segmentation_model.device)
for i,j in enumerate(all_vocab):
    if j.startswith("##"):
        TOKENS_STARTING_WITH_HASH[i]=True


segmentation_classifier = pipeline("ner", model=segmentation_model, tokenizer=segmentation_tokenizer, device=int_segmentation_device)

torch.no_grad()

bokmal_label="B"
nynorsk_label="N"

bokmal_label_id=1
nynorsk_label_id=2

model_config=json.load(open("./models/sentence_segmentation/config.json","r"))
ID2LABEL=model_config["id2label"]
ID2LABEL={i:"bm" if ID2LABEL[i]==bokmal_label else "nn" if ID2LABEL[i]==nynorsk_label else "" for i in ID2LABEL}


def tag(text , given_lang="au"):
    global segmentation_classifier
    global segmentation_tokenizer
    global segmentation_device
    global segmentation_model
    global classification_model
    global ID2LABEL

    global MODEL_SENTENCE_START_ID
    global MODEL_SENTENCE_END_ID
    global MAX_LENGTH_WITHOUT_CLS
    global BATCH_SIZE

    global bokmal_label
    global nynorsk_label

    global bokmal_label_id
    global nynorsk_label_id

    global TOKENS_STARTING_WITH_HASH


    global class_to_label_nn
    global class_to_label_bm

    # Here we get the whole text tokenized.
    text=text.replace("\n", " ")
    encodings = segmentation_tokenizer(text,add_special_tokens=False, return_tensors="pt").to(segmentation_model.device)
    
    # Save a copy of the tokenization
    original_encodings=copy.deepcopy(encodings)

    # Pad to the complete size (model max_size -1 (-1 to add CLS))
    old_size=encodings["input_ids"][0].size()[0]
    
    # Pad size
    pad_size=MAX_LENGTH_WITHOUT_CLS - old_size%MAX_LENGTH_WITHOUT_CLS

    # Number of rows
    row_count=int(old_size/MAX_LENGTH_WITHOUT_CLS) + 1

    # Do padding with Zeros to the pad_size that we have calculated.
    encodings["input_ids"] = torch.nn.functional.pad(input=encodings["input_ids"], pad=(0, pad_size), mode='constant', value=0)

    # Set the last token as SENTENCE END (SEP)
    encodings["input_ids"][0][old_size]=MODEL_SENTENCE_END_ID
    
    # Chunk into max_length items
    encodings["input_ids"]=torch.reshape(encodings["input_ids"],(row_count,MAX_LENGTH_WITHOUT_CLS))

    # Add CLS to each item
    encodings["input_ids"]=torch.cat(( torch.full((row_count,1),MODEL_SENTENCE_START_ID, device=segmentation_model.device) ,encodings["input_ids"]),dim=1)

    # Create attention mask and token_type_ids
    encodings["attention_mask"]=torch.ones_like(encodings["input_ids"], device=segmentation_model.device)
    encodings["token_type_ids"]=torch.zeros_like(encodings["input_ids"], device=segmentation_model.device)

    # Create batches
    input_ids_batched=torch.split(encodings["input_ids"], BATCH_SIZE)
    attention_mask_batched=torch.split(encodings["attention_mask"], BATCH_SIZE)
    token_type_ids_batched=torch.split(encodings["token_type_ids"], BATCH_SIZE)
   
    # Set the last chunk's attention mask according to its size 
    attention_mask_batched[-1][-1][pad_size +1:] = 0

    # Now pass all chunks through the model and get the labels
    # While passing, we count the number of bokmal and nynorsk markers
    labels_output=[]
    labels_ids=[0,0,0]
    for input_ids, attention_masks, token_type_ids in zip(input_ids_batched, attention_mask_batched, token_type_ids_batched):
        current_batch={"input_ids":input_ids, "attention_mask":attention_masks, "token_type_ids":token_type_ids}
        outputs = segmentation_model(**current_batch)
        label_data=outputs.logits.argmax(-1)
        label_counts_in_this_chunk=label_data.unique(return_counts=True)
        for l_id, num in zip(label_counts_in_this_chunk[0].tolist(), label_counts_in_this_chunk[1].tolist() ):
            if l_id!=0:
                labels_ids[l_id]+=num
        labels_output.extend(label_data)

    # Determine the languge. If the language is given as parameter use that.
    # If not, use labels_ids to determine the language
    if given_lang=="au" or given_lang==None:
        if labels_ids[1]>labels_ids[2]:
            class_to_label=class_to_label_nn
        else:
            class_to_label=class_to_label_bm
    elif given_lang=="bm":
        class_to_label=class_to_label_bm
    else:
        class_to_label=class_to_label_nn
    
    # Serialize back 
    labels_output=torch.stack(labels_output ,dim=0)
    labels_output=labels_output[:, range(1,MAX_LENGTH_WITHOUT_CLS+1)]
    labels_output=torch.reshape(labels_output,(1,row_count *MAX_LENGTH_WITHOUT_CLS))

    # Now the data is split into sentences
    # So, now create sentence data as list so that this could be used 
    # in torch operations and can be input to the models
    sentence_list=[]
    this_sentence=[MODEL_SENTENCE_START_ID]
    last_label=-1
    for token, label in zip(original_encodings["input_ids"][0].tolist(), labels_output[0].tolist()):
        if TOKENS_STARTING_WITH_HASH[token]:
            if last_label!=0:
                sentence_list[-1].append(token)
            else:
                this_sentence.append(token)
                last_label=label
        elif label==0:
            this_sentence.append(token)
            last_label=label
        else:
            this_sentence.append(token)
            sentence_list.append(this_sentence)
            this_sentence=[MODEL_SENTENCE_START_ID]
            last_label=label

    # Remove any tensors from the GPU since we have sentences in the memory now
    del original_encodings
    del labels_output
    del attention_mask_batched
    del input_ids_batched
    del token_type_ids_batched
    del encodings
    del old_size
    del text
    del outputs
    torch.cuda.empty_cache()

    # Create batches
    batched_sentences=[]
    my_batch=[]
    num_sentences=0
    for sentence in sentence_list:
        sentence.append(MODEL_SENTENCE_END_ID)
        if num_sentences==BATCH_SIZE:
            max_len=len(max(my_batch, key=len))
            if max_len>segmentation_tokenizer.model_max_length:
                max_len=segmentation_tokenizer.model_max_length
            
            my_attentions=torch.LongTensor([[1] * len(i[0:max_len]) + [0]*(max_len-len(i[0:max_len])) for i in my_batch]).to(classification_model.device)
            my_batch=[i[0:max_len] + [0]*(max_len-len(i[0:max_len])) for i in my_batch]
            to_append={
                                    "input_ids": torch.LongTensor(my_batch).to(classification_model.device),
                                    "attention_mask": my_attentions,
                                    "token_type_ids": torch.zeros_like(my_attentions, device=classification_model.device)
                                    }
            batched_sentences.append(to_append)
            num_sentences =0
            my_batch=[]
        else:
            my_batch.append(sentence)
            num_sentences+=1

    max_len=len(max(my_batch, key=len))
    my_attentions=torch.LongTensor([[1] * len(i[0:max_len]) + [0]*(max_len-len(i[0:max_len])) for i in my_batch]).to(classification_model.device)
    my_batch=[i[0:max_len] + [0]*(max_len-len(i[0:max_len])) for i in my_batch]
    to_append={
                            "input_ids": torch.LongTensor(my_batch).to(classification_model.device),
                            "attention_mask": my_attentions,
                            "token_type_ids": torch.zeros_like(my_attentions, device=classification_model.device)
                            }
    batched_sentences.append(to_append)

    # Now use the classification model to tag
    # and tokenization model to merge tokens
    for my_batch in batched_sentences:
        outputs = classification_model(**my_batch)
        classification_output=outputs.logits.argmax(-1)
        if classification_model.device!=tokenization_model.device:
            my_batch["input_ids"]=my_batch["input_ids"].to(tokenization_model.device)
            my_batch["attention_mask"]=my_batch["attention_mask"].to(tokenization_model.device)
            my_batch["token_type_ids"]=my_batch["token_type_ids"].to(tokenization_model.device)
        outputs = tokenization_model(**my_batch)
        tokenization_output=outputs.logits.argmax(-1)

        for i in range(int(classification_output.size()[0])):            
            classes = [class_to_label[ classification_model.config.id2label[t.item()] ] if classification_model.config.id2label[t.item()] in class_to_label else "" for t in classification_output[i]]
            tag=[]
            for j,k,l in zip(my_batch["input_ids"][i], tokenization_output[i], classes):
                if j==MODEL_SENTENCE_START_ID:
                    continue
                elif j==MODEL_SENTENCE_END_ID:
                    break

                if TOKENS_STARTING_WITH_HASH[j]:
                    tag[-1]["w"] += segmentation_tokenizer.decode(j)[2:]
                elif k==0:
                    if len(tag)>0:
                        tag[-1]["w"] += segmentation_tokenizer.decode(j)
                    else:
                        tag.append({"w":segmentation_tokenizer.decode(j) , "t":l})
                else:
                    tag.append({"w":segmentation_tokenizer.decode(j) , "t":l})

            print(json.dumps(tag))

def main():
    global BATCH_SIZE
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", dest="filename",
                    help="file to process", metavar="FILE")
    parser.add_argument("-bm", dest='spraak', action='store_const', const='bm', default='au',
                    help='Tag Bokmål')
    parser.add_argument("-nn", dest='spraak', action='store_const', const='nn', default='au',
                    help='Tag Nynorsk')
    parser.add_argument("-au", dest='spraak', action='store_const', const='au', default='au',
                    help='Identify the langauge (default)')
    parser.add_argument('-b','--batch-size', action="store", default="8",type=str, required=False, help='Batch size for the GPU processing.')

    args = parser.parse_args()

    if args.batch_size is not None:
        try:
            BATCH_SIZE=int(args.batch_size)
        except:
            pass
 
    if args.filename is not None:
        if os.path.isfile(args.filename):
            tag(open(args.filename,"r").read().strip().replace("\r",""), args.spraak ) #.split(line_separator)
            
        else:
            print("The file " + args.filename + " could not be found.")
            exit(1)

if __name__ == '__main__':
    main()
