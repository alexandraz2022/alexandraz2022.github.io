# Using Natural Language Procesing for Aerospace Intelligence

# Colombian Aeroespace Force, Universidad de los Andes, Florida State University

#### Deep Learning Named Entity Recognition Models 

## Abstract
An empirical study conducted on three datasets related to the Colombian
Aerospace Force, aiming to evaluate the performance of three deep learning models in extracting
entities from documents written in Spanish. We used models with transformers and convolutional
architectures to compare their performance in the Precision, Recall, and F1 Metrics. Our
findings indicate that the models under study, which were initially designed for NER tasks,
exhibited limitations when applied off-the-shelf. However, after fine-tuning, their performance
increased considerably, making them useful in military contexts with aerospace technology for
extracting valuable information from large volumes of data.


### BERT
[Dataset 1](https://thesoftwaredesignlab.github.io)
[Dataset 2](https://thesoftwaredesignlab.github.io)

The Bidirectional Encoder Representations from Transformers approach(BERT)[8] is one of the most used in the last few years accounting for a total of 70+ NER
models capable of predicting text over 70 languages. BERT was designed by Google researchers and it is based on the **Transformer architecture**** based solely on attention
mechanisms on long sequences of text to identify their connections[9].

!(/assets/images/bert.png)

### Flair
[Dataset 1](https://thesoftwaredesignlab.github.io)
[Dataset 2](https://thesoftwaredesignlab.github.io)

NLP framework developed bythe Humboldt University of Berlin, known as Flair. This framework integrates NER models in four languages: English, Dutch, **Spanish**, and German[18].

!(/assets/images/bert.png)

### spaCy
[Dataset 1](https://thesoftwaredesignlab.github.io)
[Dataset 2](https://thesoftwaredesignlab.github.io)

Open-source software library that allows for the extraction of information from large volumes of data using Natural Language Processing in 26 different languages. It was written in the programming
language Python, and one of its functions is focused on extracting entities[22]. SpaCy uses a neural network model within a transition-based parser model, incorporating Convolutional Neural Network (CNN) encoding layers to reduce the word dimensionality along three subnetworks.

!(/assets/images/bert.png)

## Models off-the-shelf

### BERT
[Dataset 2](https://thesoftwaredesignlab.github.io)

### FLAIR
[Dataset 2](https://thesoftwaredesignlab.github.io)

### spaCy
[Dataset 2](https://thesoftwaredesignlab.github.io)

## Datasets
**First dataset (𝐷𝑆1) **
-  (𝐷𝑆1)  using public documents or academic literature available on the internet and published by either the official communications of the Colombian Ministry of Defense, the General Command of
Colombian Military Forces, or by military personnel from these institutions. This dataset includes paragraphs of awareness situations, events related to security breaches, or acts of violence in the Spanish language, which include military argot and describe historical facts. 

**Second dataset (𝐷𝑆1) **
- (𝐷𝑆2) focuses on one of the most crucial areas of concern for the Colombian Aerospace Force. It
includes 187 news spanning from 2013 to 2023, delving into events related to the Colombian Amazon Affectations. This
analysis of data is important because is one source of data to complement the information obtained using aerial missions,
satellite sensors, infrared optics sensors, and specialized software processing, among other aerospace technologies
to analyze the media environment impact focus on the Colombian Amazon Region. In this dataset, we can find
information about the entities involved or affected by these events. It was created by students from the Noncommissioned
Officer Academy in the Colombian Aerospace Force and was used in the programming marathon dedicated to Amazon protection, known as Codefest AD ASTRA 2023 [ 51 ]. 

## Models 

## References
- Causality: The new science of an old question - GSP Seminar, Fall 2021
- Guest Lecture: Dimensionality Reduction - Big Data and Machine Learning for Scientific Discovery (PHYS 5336), Spring 2021
- Guest Lecture: Fourier and Wavelet Transforms - Scientific Computing (PHYS 5315), Fall 2020
- A Brief Introduction to Optimization - GSP Seminar, Fall 2019
- Weeks of Welcome Poster Competition - UTD, Fall 2019
- A Brief Introduction to Networks - GSP Seminar, Spring 2019


- [Data Science Blog](https://medium.com/@shawhin)
