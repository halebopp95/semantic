import spacy
nlp = spacy.load('en_core_web_md')

# CODE EXTRACT 1---------------------------------
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

''' 
The highest output is for cat and monkey, second highest for banana and monkey, and least for cat and banana.
This tracks  with how I would have classified similarity - cat and monkey are both mammals, so the same kind 
of animal. Monkey and banana are different types of things but related because monkeys eat bananas. Bananas
and cats are relatively unrelated.

If I was to compare car, gun and sword, I would expect gun and sword to have the highest similarity as they
are both weapons, followed by car and gun, as they are both machines, and finally car and sword.'''

word1 = nlp("car")
word2 = nlp("gun")
word3 = nlp("sword")

print(word1.similarity(word2)) # 0.340
print(word3.similarity(word2)) # 0.370
print(word3.similarity(word1)) # 0.099


# CODE EXTACT 2 ---------------------------------------------
tokens = nlp('cat apple monkey banana')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))


# CODE EXTRACT 3 ---------------------------------------------
sentence_to_compare = "Why is my cat on the car"

sentences = ["Where did my dog go",
"Hello, there is my car",
"I\'ve lost my cat",
"I\'d like my boat back",
"I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

# EXAMPLE FILE COMMENTS ----------------------------------------------

''' 
When I run the example file with the simpler language module 'en_core_web_sm I get the following
warning message:

UserWarning: [W007] The model you're using has no word vectors loaded, so the result of the Doc.similarity 
method will be based on the tagger, parser and NER, which may not give useful similarity judgements. 
This may happen if you're using one of the small models, e.g. `en_core_web_sm`, which don't ship with word vectors 
and only use context-sensitive tensors. You can always add your own word vectors, or use one of the larger models 
instead if available.

'I still receive results from the similarity methods, but the numbers are a lot lower (an average of .5 or .6 instead of .8)'''