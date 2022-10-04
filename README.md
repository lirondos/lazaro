# Observatorio Lázaro
This is the code repository of Observatorio Lázaro, an observatory of anglicism usage in the Spanish press. The purpose of this project is to apply a data-driven approach to the study of anglicisms (ie, unadapted lexical borrowings from English) in Spanish newspapers. Every day, Observatorio Lázaro collects the latests news published in 22 Spanish news sources, analyzes them and extracts the anglicisms that have been used in the daily news.

The core of the project is a Machine Learning model that extracts unadapted lexical borrowings (especially English lexical borrowings or *anglicisms*) from Spanish articles. The model is a BiLSTM-CRF model fed with word and subword embeddings. More information on the model can be found in the paper [*Detecting Unassimilated Borrowings in Spanish: An Annotated Corpus and Approaches to Modeling*](https://aclanthology.org/2022.acl-long.268/). More info on the motivation behind the project can be found at the [*About* section in Observatorio Lázaro website](https://observatoriolazaro.es/en/acerca.html).

The name of this project, Lázaro, is an homage to Spanish philologist [Fernando Lázaro Carreter](https://es.wikipedia.org/wiki/Fernando_L%C3%A1zaro_Carreter), whose columns admonishing against the usage of anglicisms on the Spanish press became extremely popular during the decades of 1980s and 1990s. 

## Observatorio Lázaro website
The output of Observatorio Lázaro, along with graphs, visualizations and aggregated info on each anglicism registered by Lázaro can be seen at [Observatorio Lázaro website](https://observatoriolazaro.es/).

## Python library and models
* The anglicism detection model that runs behind Observatorio Lázaro is available through the [Python library ``pylazaro``](https://pylazaro.readthedocs.io/) or through [HuggingFace model hub](https://huggingface.co/models?other=arxiv:2203.16169).
* The dataset used to train the model behind the Observatory is the [COALAS corpus](https://github.com/lirondos/coalas>).

## Previous versions
A previous version of the Observatorio ran on a CRF model and tracked 8 Spanish newspapers. 
* For the code of that old version see [``crf-aug2022`` branch](https://github.com/lirondos/lazaro/tree/crf-aug2022).
* For the technical creation behind the Observatorio see my MS thesis [*Lázaro: An Extractor of Emergent Anglicisms in Spanish Newswire*](https://scholarworks.brandeis.edu/esploro/outputs/9923880179101921).


## Citation
If you use the Observatory, please cite the following references:
```
@inproceedings{alvarez-mellado-lignos-2022-detecting,
    title = "Detecting Unassimilated Borrowings in {S}panish: {A}n Annotated Corpus and Approaches to Modeling",
    author = "{\'A}lvarez-Mellado, Elena  and
      Lignos, Constantine",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.268",
    pages = "3868--3888",
    abstract = "This work presents a new resource for borrowing identification and analyzes the performance and errors of several models on this task. We introduce a new annotated corpus of Spanish newswire rich in unassimilated lexical borrowings{---}words from one language that are introduced into another without orthographic adaptation{---}and use it to evaluate how several sequence labeling models (CRF, BiLSTM-CRF, and Transformer-based models) perform. The corpus contains 370,000 tokens and is larger, more borrowing-dense, OOV-rich, and topic-varied than previous corpora available for this task. Our results show that a BiLSTM-CRF model fed with subword embeddings along with either Transformer-based embeddings pretrained on codeswitched data or a combination of contextualized word embeddings outperforms results obtained by a multilingual BERT-based model.",
}
```
```
@masterthesis{ÁlvarezMelladoElena2020LAEo,
title = {Lázaro: An Extractor of Emergent Anglicisms in Spanish Newswire},
abstract = {The use of lexical borrowings from English (often called anglicisms) in the Spanish press evokes great interest, both in the Hispanic linguistics community and among the general public. Anglicism usage in Spanish language has been previously studied within the field of corpus linguistics. Prior work has traditionally relied on manual inspection of corpora, with the limitations that implies. This thesis proposes a model for automatic extraction of unadapted anglicisms in Spanish newswire. This thesis introduces: (1) an annotated corpus of 21,570 newspaper headlines (325,665 tokens) written in European Spanish annotated with unadapted anglicisms and (2) two sequencelabeling models to perform automatic extraction of unadapted anglicisms: a conditional random field model with handcrafted features and a BiLSTM-CRF model with word and character embeddings. The best results are obtained by the CRF model, with an F1 score of 89.60 on the development set and 87.82 on the test set. Finally, a practical application of the CRF model is presented: an automatic pipeline that performs daily extraction of anglicisms from the main national newspapers of Spain.},
author = {Álvarez Mellado, Elena},
keywords = {anglicism detection;lexical borrowing;Spanish newswire},
language = {eng},
school = {Brandeis University, Graduate School of Arts and Sciences},
year = {2020},
}
```
