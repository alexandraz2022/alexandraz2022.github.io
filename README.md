# Using Natural Language Processing for Aerospace Intelligence

# Colombian Aeroespace Force, Universidad de los Andes, Florida State University

#### Deep Learning Named Entity Recognition Models 


An empirical study conducted on three datasets related to the Colombian
Aerospace Force, aims to evaluate the performance of three deep learning models in extracting
entities from documents written in Spanish. We used models with transformers and convolutional
architectures to compare their performance in the Precision, Recall, and F1 Metrics. Our
findings indicate that the models under study, which were initially designed for NER tasks,
exhibited limitations when applied off-the-shelf. However, after fine-tuning, their performance
increased considerably, making them useful in military contexts with aerospace technology for
extracting valuable information from large volumes of data.

## Models Fine-tuned in aerospace intelligence context

### BERT
[FTBERT_DS1](https://fuerzaaereacolombia-my.sharepoint.com/:f:/g/personal/alexandra_zabala_fac_mil_co/EuOZ89G_HtBPj_a6zvht3awB_nvkdXQmWJ0i0TKcizDweg?e=cwhDen)
[FTBERT_DS2](https://fuerzaaereacolombia-my.sharepoint.com/:f:/g/personal/alexandra_zabala_fac_mil_co/EuOZ89G_HtBPj_a6zvht3awB_nvkdXQmWJ0i0TKcizDweg?e=cwhDen)

The Bidirectional Encoder Representations from Transformers approach(BERT)[1] is one of the most used in the last few years accounting for a total of 70+ NER
models capable of predicting text over 70 languages. BERT was designed by Google researchers and it is based on the **Transformer architecture**** based solely on attention
mechanisms on long sequences of text to identify their connections[2].

![Prueba](/assets/img/bert.png)

### Flair
[FTflair_DS1](https://fuerzaaereacolombia-my.sharepoint.com/:f:/g/personal/alexandra_zabala_fac_mil_co/EuOZ89G_HtBPj_a6zvht3awB_nvkdXQmWJ0i0TKcizDweg?e=cwhDen)
[FTflair_DS2](https://fuerzaaereacolombia-my.sharepoint.com/:f:/g/personal/alexandra_zabala_fac_mil_co/EuOZ89G_HtBPj_a6zvht3awB_nvkdXQmWJ0i0TKcizDweg?e=cwhDen)

NLP framework developed bythe Humboldt University of Berlin, known as Flair. This framework integrates NER models in four languages: English, Dutch, **Spanish**, and German[3][4].

![Prueba](/assets/img/flair.jpg)

### spaCy
[FTspaCy_DS1](https://fuerzaaereacolombia-my.sharepoint.com/:f:/g/personal/alexandra_zabala_fac_mil_co/EuOZ89G_HtBPj_a6zvht3awB_nvkdXQmWJ0i0TKcizDweg?e=cwhDen)
[FTspaCy DS2](https://fuerzaaereacolombia-my.sharepoint.com/:f:/g/personal/alexandra_zabala_fac_mil_co/EuOZ89G_HtBPj_a6zvht3awB_nvkdXQmWJ0i0TKcizDweg?e=cwhDen)

spaCy is a Open-source software library that allows for the extraction of information from large volumes of data using Natural Language Processing in 26 different languages. SpaCy uses a neural network model within a transition-based parser model, incorporating **Convolutional Neural Network (CNN)** encoding layers to reduce the word dimensionality along three subnetworks[5].

![Prueba](/assets/img/spacy.jpg)

## Datasets
We have created a corpus of 694 documents across three datasets. The first dataset (𝐷𝑆1) contains historical events from public sources such as the Military Historical Memory Report in 2019 [6] or the Executive Report on Achievements and Mission Challenges of the Colombian Defense Sector in 2021 [7], among others; the second (𝐷𝑆2) contains news related to the Colombian Amazon searched through google; and the third dataset (𝐷𝑆3) consists of classified data containing internal textual reports from the Colombian Aerospace Force. 

-  **(𝐷𝑆1)** Using public documents or academic literature available on the internet and published by either the official communications of the Colombian Ministry of Defense, the General Command of
Colombian Military Forces, or by military personnel from these institutions. This dataset includes paragraphs of awareness situations, events related to security breaches, or acts of violence in the Spanish language, which include military argot and describe historical facts.
[Dataset 1](https://github.com/alexandraz2022/alexandraz2022.github.io/blob/main/datasets)

-  **(𝐷𝑆2)** Focusing on one of the most crucial areas of concern for the Colombian Aerospace Force. DS2 includes 187 news spanning from 2013 to 2023, delving into events related to the Colombian Amazon Affectations. This data is one source of data to complement the information obtained using aerial missions, satellite sensors, infrared optics sensors, and specialized software processing, among other aerospace technologies to analyze the media environment impact focus on the Colombian Amazon Region. In this dataset, we can find information about the entities involved or affected by these events. It was created by students from the Noncommissioned Officer Academy in the Colombian Aerospace Force and was used in the programming marathon dedicated to Amazon protection, known as Codefest AD ASTRA 2023[8].
[Dataset 2](https://github.com/alexandraz2022/alexandraz2022.github.io/blob/main/datasets)


## Models off-the-shelf

### BERT
For the case of BERT models, we used BETO cased NER fine-tuned model (BETO CFT), it is a large Spanish corpus and fine-tuned specifically to conduct NER TASK in Spanish

[BETO CFT](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased-finetuned-ner/commit/0cf7cc10bc005707fa8a70ba3739c7d1b50b2630)

### FLAIR
We used Spanish-NER-Flair-large-model (SFLM); this model uses a transformer architecture and has not been documented in a military context in the literature; however, its performance in the Spanish language was assessed during the open innovation event CODEFEST AD ASTRA 2023 [8]

[SFLM](https://huggingface.co/flair/ner-spanish-large)

### spaCy
The model used by spaCy library was es-core-news-lg model (ECNLM), it features a data pipeline with NER components that achieved the highest scores among the three available spaCy models for the Spanish language 

[ECNLM](https://spacy.io/models/es)


## References
[1]Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K., “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,” Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), Association for Computational Linguistics, Minneapolis, Minnesota, 2019, pp. 4171–4186. https://doi.org/10.18653/v1/N19-1423, URL https://aclanthology.org/N19-1423.

[2]Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L. u., and Polosukhin, I., “Attention is All you Need,” Advances in Neural Information Processing Systems, Vol. 30, edited by I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, Curran Associates, Inc., 2017.

[3]Akbik, A., Blythe, D., and Vollgraf, R., “Contextual String Embeddings for Sequence Labeling,” COLING 2018, 27th International Conference on Computational Linguistics, 2018, pp. 1638–1649.

[4]Schweter, S., and Akbik, A., “FLERT: Document-Level Features for Named Entity Recognition,” CoRR, Vol. abs/2011.06993, 2020. URL https://arxiv.org/abs/2011.06993

[5] Honnibal, M., and Montani, I., “spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing,” 2017. 

[6]Coral, E. J. A., and Cely, E. G., “La construcción de memoria histórica militar como aporte en la construcción de la verdad en Colombia,” Estudios en Seguridad y Defensa, Vol. 14, No. 28, 2019, pp. 307–328 

[7]Fuerzas Militares de Colombia, “Informe Ejecutivo Logros y Retos Misionales Vigencia 2021,” 2021.

[8]Fuerza Aérea Colombiana, U. d. l. A., “Codefest 2023,” , 2023. URL https://sistemas.uniandes.edu.co/codefest/2023/.



- [The Software Design Lab (TSDL)](https://thesoftwaredesignlab.github.io/)
