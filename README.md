# lazaro
An annotated corpus and CRF model for automatic extraction of anglicisms in Spanish newswire. 

This repo contains:
1. A corpus of 21,570 newspaper headlines written in European Spanish annotated with emergent anglicisms (see ```data``` folder). The headlines were extracted from the Spanish newspaper [eldiario.es](https://www.eldiario.es/).
2. A Conditional Random Field baseline model for anglicism extraction.

* ```baseline.py``` runs the model. 
* ```utils.py``` y ```utils2.py``` contain the auxiliary classes for the model (encoder, feature extractor, etc).
* ```data``` contains the corpus: training set, dev set, test set and supplemental test set. 

For more information please check: 
*An Annotated Corpus of Emerging Anglicisms in Spanish Newspaper Headlines*
Paper to be presented at the 4th Workshop on Computational Approaches to Linguistic Code-Switching (colocated with LREC 2020) https://arxiv.org/abs/2004.02929

