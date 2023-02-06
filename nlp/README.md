<div align="center">

  <img src="images/images/nlp.png" width="25%">

  
 📚  📝 🎥 NLP resources  🎥 📝 📚  
   
![Version 0.0.1](https://img.shields.io/badge/Version-0.0.1-blue.svg)
</div>

## Directory Tree

```bash
.
├── imeages
│   ├── nlp.png
├── .gitignore
├── README.md
```

1 directories, 3 files

## Natural Language Processing

Natural Language Processing (NLP) is the sub-part of Artificial Intelligence that explores how machines interact with human language. 

Part computer science, part linguistics, part statistics — it can be a challenge deciding where to begin.

Books and online courses are a great place to start!


Online Course/Book                   | Difficulty Level 
------------------------- | ---------------
[Michigan Introduction to NLP🎥][michigannlp]  | ![Introductory](https://img.shields.io/badge/Level-Introductory-brightgreen.svg)
[Jurafsky and Manning Introduction to Natural Language Processing🎥][jurafskynlp]| ![Introductory](https://img.shields.io/badge/Level-Introductory-brightgreen.svg)
[NLP - Natural Language Processing with Python🎥][udemynlp]| ![Introductory](https://img.shields.io/badge/Level-Introductory-brightgreen.svg)
[Modern Natural Language Processing in Python🎥][udemynlp]| ![Intermediate](https://img.shields.io/badge/Level-Introductory-brightgreen.svg)
[cs224n Natural Language Processing with Deep Learning GOLDEN 2019🎥][stanfordnlp2019] |![Intermediate](https://img.shields.io/badge/Level-Intermediate-yellow.svg)
[cs224u Natural Language Understanding 2019 🎥][stanfordnlu] |![Intermediate](https://img.shields.io/badge/Level-Intermediate-yellow.svg)
[cmu 2021 Neural Nets for NLP 2021🎥][cmunlp2021]|![Intermediate](https://img.shields.io/badge/Level-Intermediate-yellow.svg)
[Oxford Natural Language Processing with Deep Learning 2017🎥][oxfordnlp] |![Intermediate](https://img.shields.io/badge/Level-Intermediate-yellow.svg)
[Jurafsky Speech and Language Processing 📚][jurafskybook]|![Intermediate](https://img.shields.io/badge/Level-Intermediate-yellow.svg)
[Christopher Manning Foundations of Statistical NLP📚][fsnlp]| ![Advanced](https://img.shields.io/badge/Level-Advanced-red.svg)
[Christopher Manning Introduction to Information Retrieval📚][manninginformationr]| ![Advanced](https://img.shields.io/badge/Level-Advanced-red.svg)


### Academic NLP Papers :
![Advanced](https://img.shields.io/badge/Level-Advanced-red.svg)

Books and online courses are a great place to start, but at some point it becomes necessary to dig deeper, 
and that means looking at the academic literature!

#### Clustering & Word/Sentence Embeddings  📝

A crucial component in most natural language processing (NLP) applications is finding an expressive 
representation for text. 

A word embedding is a vector representation of a word. Words, which often occur in similar context (a king and a queen),
are assigned vector that are close by each other and words, which rarely occur in a similar context 
(a king and a motorcycle), dissimilar vectors. The embedding vectors are dense, 
relatively low dimensional (typically 50-300 dimensions) vectors.

Modern methods are typically based on sentence embeddings that map a 
sentence onto a numerical vector. The vector attempts to capture the semantic content of the text. 
If two sentences express a similar idea using different words, their representations 
(embedding vectors) should still be similar to each other.

- Peter F Brown, et al.: Class-Based n-gram Models of Natural Language, 1992.

- Tomas Mikolov, et al.: Efficient Estimation of Word Representations in Vector Space, 2013.

- Tomas Mikolov, et al.: Distributed Representations of Words and Phrases and their Compositionality, NIPS 2013.

- Quoc V. Le and Tomas Mikolov: Distributed Representations of Sentences and Documents, 2014.

- Jeffrey Pennington, et al.: GloVe: Global Vectors for Word Representation, 2014.

- Ryan Kiros, et al.: Skip-Thought Vectors, 2015.

- Wieting et al.: Towards universal paraphrastic sentence embeddings, 2015.

- Adi et al.: Fine-grained Analysis of Sentence Embeddings Using Auxiliary Prediction Tasks, 2016.

- Conneau et al.: Supervised Learning of Universal Sentence Representations from Natural Language Inference Data (InferSent), 2017.

- Piotr Bojanowski, et al.: Enriching Word Vectors with Subword Information, 2017.

- Daniel Cer et al.: [Universal Sentence Encoder](https://arxiv.org/abs/1803.11175), 2018.

#### Topic Models  📝

Topic modeling is a method in natural language processing (NLP) used to train machine learning models. It refers to the 
process of logically selecting words that belong to a certain topic from within a document.

- Thomas Hofmann: Probabilistic Latent Semantic Indexing, SIGIR 1999.

- David Blei, Andrew Y. Ng, and Michael I. Jordan: Latent Dirichlet Allocation, J. Machine Learning Research, 2003.

#### Language Modeling  📝

Language modeling is the task of predicting the next word or character in a document.

- Joshua Goodman: A bit of progress in language modeling, MSR Technical Report, 2001.

- Stanley F. Chen and Joshua Goodman: An Empirical Study of Smoothing Techniques for Language Modeling, ACL 2006.

- Yee Whye Teh: A Hierarchical Bayesian Language Model based on Pitman-Yor Processes, COLING/ACL 2006.

- Yee Whye Teh: A Bayesian interpretation of Interpolated Kneser-Ney, 2006.

- Yoshua Bengio, et al.: A Neural Probabilistic Language Model, J. of Machine Learning Research, 2003.

- Andrej Karpathy: The Unreasonable Effectiveness of Recurrent Neural Networks, 2015.

- Yoon Kim, et al.: Character-Aware Neural Language Models, 2015.

- Alec Radford, et al.: [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), 2018.

- Peters et al.: Deep contextualized word representations(ELMo), 2018.

- Howard and Ruder: Universal language model fine-tuning for text classification(ULMFit), 2018.

- Radford et al.: Improving language understanding by generative pre-training, 2018.

- Devlin et al.: Bert: Pre-training of deep bidirectional transformers for language understanding, 2018.

#### Segmentation, Tagging, Parsing  📝

Text segmentation is the process of dividing written text into meaningful units, such as words, sentences, or topics. 

Part-of-speech tagging (POS tagging) is the task of tagging a word in a text with its part of speech. A part of speech 
is a category of words with similar grammatical properties. Common English parts of speech are noun, verb, adjective, 
adverb, pronoun, preposition, conjunction, etc.

Dependency parsing is the task of extracting a dependency parse of a sentence that represents its grammatical structure 
and defines the relationships between "head" words and words, which modify those heads.

- Donald Hindle and Mats Rooth. Structural Ambiguity and Lexical Relations, Computational Linguistics, 1993.

- Adwait Ratnaparkhi: A Maximum Entropy Model for Part-Of-Speech Tagging, EMNLP 1996.

- Eugene Charniak: A Maximum-Entropy-Inspired Parser, NAACL 2000.

- Michael Collins: Discriminative Training Methods for Hidden Markov Models: Theory and Experiments with Perceptron Algorithms, EMNLP 2002.

- Dan Klein and Christopher Manning: Accurate Unlexicalized Parsing, ACL 2003.

- Dan Klein and Christopher Manning: Corpus-Based Induction of Syntactic Structure: Models of Dependency and Constituency, ACL 2004.

- Joakim Nivre and Mario Scholz: Deterministic Dependency Parsing of English Text, COLING 2004.

- Ryan McDonald et al.: Non-Projective Dependency Parsing using Spanning-Tree Algorithms, EMNLP 2005.

- Daniel Andor et al.: Globally Normalized Transition-Based Neural Networks, 2016.

- Oriol Vinyals, et al.: Grammar as a Foreign Language, 2015.

#### Sequential Labeling & Information Extraction  📝
Semantic role labeling aims to model the predicate-argument structure of a sentence and is often described as answering 
"Who did what to whom". BIO notation is typically used for semantic role labeling.

Information Extraction is the process of sifting through unstructured data and extracting vital information into more 
editable and structured data forms is known as information extraction.

- Marti A. Hearst: Automatic Acquisition of Hyponyms from Large Text Corpora, COLING 1992.

- Collins and Singer: Unsupervised Models for Named Entity Classification, EMNLP 1999.

- Patrick Pantel and Dekang Lin, Discovering Word Senses from Text, SIGKDD, 2002.

- Mike Mintz et al.: Distant supervision for relation extraction without labeled data, ACL 2009.

- Zhiheng Huang et al.: Bidirectional LSTM-CRF Models for Sequence Tagging, 2015.

- Xuezhe Ma and Eduard Hovy: End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF, ACL 2016.

#### Machine Translation & Transliteration 📝

Machine translation is the task of translating a sentence in a source language to a different target language.

Transliteration is a mechanism for converting a word in a source (foreign) language to a target 
language, and often adopts approaches from machine translation. 

In machine translation,  the objective is to preserve the semantic meaning of the utterance as much as possible while 
following the syntactic structure in the target language. In Transliteration, 
the objective is to preserve the original pronunciation of the source word as much as possible while following the 
phonological structures of the target language.

- Peter F. Brown et al.: A Statistical Approach to Machine Translation, Computational Linguistics, 1990.

- Kevin Knight, Graehl Jonathan. Machine Transliteration. Computational Linguistics, 1992.

- Dekai Wu: Inversion Transduction Grammars and the Bilingual Parsing of Parallel Corpora, Computational Linguistics, 1997.

- Kevin Knight: A Statistical MT Tutorial Workbook, 1999.

- Kishore Papineni, et al.: BLEU: a Method for Automatic Evaluation of Machine Translation, ACL 2002.

- Philipp Koehn, Franz J Och, and Daniel Marcu: Statistical Phrase-Based Translation, NAACL 2003.

- Philip Resnik and Noah A. Smith: The Web as a Parallel Corpus, Computational Linguistics, 2003.

- Franz J Och and Hermann Ney: The Alignment-Template Approach to Statistical Machine Translation, Computational Linguistics, 2004.

- David Chiang. A Hierarchical Phrase-Based Model for Statistical Machine Translation, ACL 2005.

- Ilya Sutskever, Oriol Vinyals, and Quoc V. Le: Sequence to Sequence Learning with Neural Networks, NIPS 2014.

- Oriol Vinyals, Quoc Le: A Neural Conversation Model, 2015.

- Dzmitry Bahdanau, et al.: Neural Machine Translation by Jointly Learning to Align and Translate, 2014.

- Minh-Thang Luong, et al.: Effective Approaches to Attention-based Neural Machine Translation, 2015.

- Rico Sennrich et al.: Neural Machine Translation of Rare Words with Subword Units. ACL 2016.

- Yonghui Wu, et al.: Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation, 2016.

- Melvin Johnson, et al.: [Google's Multilingual Neural Machine Translation System: Enabling Zero-Shot Translation](https://arxiv.org/abs/1611.04558), 2016.

- Jonas Gehring, et al.: Convolutional Sequence to Sequence Learning, 2017.

- Ashish Vaswani, et al.: Attention Is All You Need, 2017.


#### Automatic Text Summarization  📝

Summarization is the task of producing a shorter version of one or several documents that preserves most of the input's meaning.


- Kevin Knight and Daniel Marcu: Summarization beyond sentence extraction. Artificial Intelligence 139, 2002.

- James Clarke and Mirella Lapata: Modeling Compression with Discourse Constraints. EMNLP-CONLL 2007.

- Ryan McDonald: A Study of Global Inference Algorithms in Multi-Document Summarization, ECIR 2007.

- Wen-tau Yih et al.: Multi-Document Summarization by Maximizing Informative Content-Words. IJCAI 2007.

- Alexander M Rush, et al.: A Neural Attention Model for Sentence Summarization. EMNLP 2015.

- Abigail See et al.: [Get To The Point: Summarization with Pointer-Generator Networks](https://www.aclweb.org/anthology/P17-1099/). ACL 2017.

#### Question Answering and Machine Comprehension  📝

Machine (Reading) Comprehension is the field of NLP where we teach machines to understand and answer questions using 
unstructured text.

Question answering is the task of answering a question.


- Pranav Rajpurkar et al.: SQuAD: 100,000+ Questions for Machine Comprehension of Text. EMNLP 2015.

- Minjoon Soo et al.: Bi-Directional Attention Flow for Machine Comprehension. ICLR 2015.

#### Generation 📝

Text generation is the task of generating text with the goal of appearing indistinguishable to human-written text. 
This task if more formally known as "natural language generation" in the literature.

- Jiwei Li, et al.: Deep Reinforcement Learning for Dialogue Generation, EMNLP 2016.

- Marc’Aurelio Ranzato et al.: Sequence Level Training with Recurrent Neural Networks. ICLR 2016.

- Samuel R Bowman et al.: [Generating sentences from a continuous space](https://www.aclweb.org/anthology/K16-1002/), CoNLL 2016.

- Lantao Yu, et al.: SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient, AAAI 2017.


```diff
- Suggestions and pull requests are welcome. The goal is to make this a collaborative effort to maintain an updated list of nlp resources.
```
