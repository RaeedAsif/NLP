#importing nltk and brown corpus
import nltk
from nltk.corpus import brown

#download if! NTLK brown and google universal_tagset
nltk.download('brown')
nltk.download('universal_tagset')

#retrieving tagged_sentences from brown of category 'news' and with tagset of google
tagged_sentences = brown.tagged_sents(categories='news', tagset = 'universal')
ts_29= "The jury praise the administration and operation of the Atlanta Police Department, the Fulton Tax Commissioner's Office, the Bellwood and Alpharetta prison farms , Grady, Hospital and the Fulton Health Department."

#extracting words and tags as a seperate list
#words = [[tag for word, tag in sent] for sent in tagged_sentences]
#tags = [tag for sent in tagged_sentences for word, tag in sent]

#computing frequency distribution over the tags
#from nltk.probability import FreqDist
#tagsFDist = FreqDist(tag.lower() for tag in tags)

## Hidden Markov Model:
train_data = tagged_sentences[:-500] # training set
test_data = tagged_sentences[-500:] # test set

#extracting words and tags as a seperate list
words = [word for sent in tagged_sentences for word, tag in sent]
tags = [tag for sent in tagged_sentences for word, tag in sent]

from nltk.tag import hmm
trainer = hmm.HiddenMarkovModelTrainer()
tagger = trainer.train_supervised(train_data)

def predict_hmm_model(sentence):
    return(tagger.tag(sentence.split())) #predicting tags of given sentence

''' # Test set data
temp=[]
for word in words:
    temp.append(predict_hmm_model(word))
print(temp)
'''

#print(predict_hmm_model(ts_29))

# Conditional probabilities
# get tagged sentences
tagged_words = [ ]
all_tags = [ ]

for sent in tagged_sentences:
    tagged_words.append( ("START", "START") )
    all_tags.append("START")
    for (word, tag) in sent:
        all_tags.append(tag)
        tagged_words.append( (tag, word) )
    tagged_words.append( ("END", "END") )
    all_tags.append("END")

## trasition probabilities [ P(t_{i} | t_{i-1}) = C(t_{i-1}, t_{i})/C(t_{i-1}) ]:
cfd_tags= nltk.ConditionalFreqDist(nltk.bigrams(all_tags)) # C(t_{i-1}, t_{i}):
cpd_tags = nltk.ConditionalProbDist(cfd_tags, nltk.MLEProbDist) # P(t_{i} | t_{i-1})

## emission probabilities [ P(w_{i} | t_{i}) =  C(t_{i}, w_{i}) / C(t_{i}) ]
cfd_tagwords = nltk.ConditionalFreqDist(tagged_words) # C(t_{i}, w_{i})
cpd_tagwords = nltk.ConditionalProbDist(cfd_tagwords, nltk.MLEProbDist)  # P(w_{i} | t_{i})

# viterbi algorithm for PoS tagging
import numpy as np
def viterbi(sentence):
    # Step 1. initialization step
    distinct_tags = np.array(list(set(all_tags)))
    tagslen = len(distinct_tags)

    sentlen = len(sentence)
    viterbi = np.zeros((tagslen, sentlen+1) ,dtype=float)
    backpointer = np.zeros((tagslen, sentlen+1) ,dtype=np.uint32)



    # Step 1. initialization step
    for s, tag in enumerate(distinct_tags):
        viterbi[s,0] =  cpd_tags["START"].prob(tag) * cpd_tagwords[tag].prob( sentence[0] )
        backpointer[s,0] = 0
        print("Viterbi probability V( {1} ,{0} ) = {2}".format(sentence[0],tag,  viterbi[s,0]) )
    print('============================')

    # Step 2. recursion step
    for t in range(1, sentlen):
        for s, tag in enumerate(distinct_tags):
            current_viterbi = np.zeros( tagslen ,dtype=float)
            for sprime, predtag in enumerate(distinct_tags):
                current_viterbi[sprime] = viterbi[sprime,t-1] * \
                                          cpd_tags[predtag].prob(tag) * \
                                          cpd_tagwords[tag].prob(sentence[t])
            backpointer[s,t] = np.argmax(current_viterbi)
            viterbi[s,t] = max(current_viterbi)
            print("Viterbi probability V( {1} ,{0} ) = {2}".format(sentence[t],tag,  viterbi[s,t]))

        print('============================')


    # Step 3. termination step
    current_viterbi = np.empty( tagslen ,dtype=float)
    ind_of_end = -1
    for s, tag in enumerate(distinct_tags):
        if tag == "END":
            ind_of_end  = s
        current_viterbi[s] = viterbi[s,sentlen-1] * cpd_tags[tag].prob("END")

    backpointer[ind_of_end,sentlen] = np.argmax(current_viterbi)
    viterbi[ind_of_end,sentlen] = max(current_viterbi)

    # Step 3. backtrace the path
    best_tagsequence = [ ]
    prob_tagsequence = viterbi[ind_of_end,sentlen]
    prevind  = ind_of_end
    for t in range(sentlen,0,-1):
        prevind = backpointer[prevind,t]
        best_tagsequence.append(distinct_tags[prevind])
    best_tagsequence.reverse()

    return best_tagsequence, prob_tagsequence

nltk.download('punkt')
#sentence =  nltk.word_tokenize(ts_29)
#best_tagsequence,prob_tagsequence = viterbi(sentence)
#print(best_tagsequence,prob_tagsequence )
