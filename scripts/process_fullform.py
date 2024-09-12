import sys
import pickle

if sys.argv[1]=="nn":
    from fullform_nn import fullformHash
else:
    from fullform_bm import fullformHash
IGNORE_TAGS={"normert","unormert", "<perf-part>","perf-part", "<trans1>", "<trans2>", "<intrans2>","<intrans1>"}

NN_TAGS={"$punc$","1","2","3",":<anf>:",":<komma>:",":<parentes-beg>:",":<parentes-slutt>:",":<strek>:",":adj:",":adv:",":det:",":inf-merke:",":interj:",":konj:",":prep:",":pron:",":sbu:",":subst:",":symb:",":ukjent:",":verb:","<adj>","<adv>","<dato>","<ellipse>","<kolon>","<next_token>","<ordenstal>","<perf-part>","<pres-part>","<punkt>","<romartal>","<semi>","<spm>","<st-verb>","<utrop>","akk","appell","bu","dem","eint","fem","fl","fork","forst","gen","hum","høfleg","imp","inf","komp","kvant","m/f","mask","nom","nøyt","pass","perf-part","pers","pos","poss","pres","pret","prop","refl","res","sp","sup","symb","ub","ubøy","ufl"}


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
NN_TAGS={i if i not in EQUAL_TAGS else EQUAL_TAGS[i] for i in NN_TAGS}
BM_TAGS={i if i not in NN_TO_BM else NN_TO_BM[i] for i in NN_TAGS}

def accent_count(st):
    num=0
    for i in st:
        if not i.isascii():
            num+=1
    return num



if sys.argv[1]=="nn" :
    USED_TAGS=NN_TAGS
else:
    USED_TAGS=BM_TAGS


new_dict=dict()
for i in fullformHash:
    lemmas={ j.split(" ")[0].strip("\"") : [k for k in j.split(" ")[1:] if k in USED_TAGS] for j in fullformHash[i].replace("\t","").split("\n") if j!="" }
    
    if i not in new_dict:
        new_dict[i]=dict()
    for lemma in lemmas:
        if len(lemmas[lemma])>0:
            if lemmas[lemma][0] not in new_dict[i]:
                new_dict[i][lemmas[lemma][0]]=list()
            new_dict[i][lemmas[lemma][0]].append({lemma:lemmas[lemma][1:]})

for word in new_dict:
    for typ in new_dict[word]:
        if len(new_dict[word][typ])==1:
            new_dict[word][typ]=list(new_dict[word][typ][0].keys())[0]
#            last_dict[word]=new_dict[word]
#            print(word + " " + typ + " " + list(new_dict[word][typ][0].keys())[0])
        else:
            found=True
            while found:
                found=False
                for i in range(len(new_dict[word][typ])):
                    for j in range(i+1,len(new_dict[word][typ])):
                        if (set(list(new_dict[word][typ][i].values())[0])==set(list(new_dict[word][typ][j].values())[0])):
                            lemma_1_accent_count=accent_count(list(new_dict[word][typ][i].keys())[0])
                            lemma_2_accent_count=accent_count(list(new_dict[word][typ][j].keys())[0])
                            if lemma_1_accent_count>lemma_2_accent_count:
                                del new_dict[word][typ][i]
                            else:
                                del new_dict[word][typ][j]
                            found=True
                            break
                    if found==True:
                        break
            if len(new_dict[word][typ])==1:
                new_dict[word][typ]=list(new_dict[word][typ][0].keys())[0]
for word in new_dict:
    for typ in new_dict[word]:
        if type(new_dict[word][typ])==list:
            new_dict[word][typ]=dict(pair for d in new_dict[word][typ] for pair in d.items())

if sys.argv[1]=="nn":
    with open('nn.pickle', 'wb') as handle:
        pickle.dump(new_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open('bm.pickle', 'wb') as handle:
        pickle.dump(new_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

#for word in new_dict:
#    for typ in new_dict[word]:
#        print(word + " " + typ + " " + str(new_dict[word][typ]))



#            for option in new_dict[word][typ]:
#                for lemma in option:
#                    print(word + " " + typ + " " + lemma + " " + str(option[lemma]))

