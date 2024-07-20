<p>
<div align="center">

# Transformer From Scratch [Python]
</div>
</p>

<p align="center" width="100%">
    <img src="./Transformer From Scratch/Assets/Transformer_Diagram.png" width="40%" height="40%" />
</p>

<div align="center">
    <a>
        <img src="https://img.shields.io/badge/Made%20with-Jupyter-white?style=for-the-badge&logo=Jupyter&logoColor=white">
    </a>
    <a>
        <img src="https://img.shields.io/badge/Made%20with-Python-white?style=for-the-badge&logo=Python&logoColor=white">
    </a>
</div>

<br/>

<div align="center">
    <a href="https://github.com/EstevesX10/Transformer-From-Scratch/blob/main/LICENSE">
        <img src="https://img.shields.io/github/license/EstevesX10/Transformer-From-Scratch?style=flat&logo=gitbook&logoColor=white&label=License&color=white">
    </a>
    <a href="">
        <img src="https://img.shields.io/github/repo-size/EstevesX10/Transformer-From-Scratch?style=flat&logo=googlecloudstorage&logoColor=white&logoSize=auto&label=Repository%20Size&color=white">
    </a>
    <a href="">
        <img src="https://img.shields.io/github/stars/EstevesX10/Transformer-From-Scratch?style=flat&logo=adafruit&logoColor=white&logoSize=auto&label=Stars&color=white">
    </a>
    <a href="https://github.com/EstevesX10/Transformer-From-Scratch/blob/main/DEPENDENCIES.md">
        <img src="https://img.shields.io/badge/Dependencies-DEPENDENCIES.md-white?style=flat&logo=anaconda&logoColor=white&logoSize=auto&color=white"> 
    </a>
</div>

## Project Overview

This project focuses on creating a `Transformer model` from the ground up using Python to perform `translation tasks` upon English sentences aiming to translate them into Portuguese. The Transformer model, discussed by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin in their 2017 paper **"Attention Is All You Need"** (In which this project was mainly based on),  has transformed the field of Natural Language Processing (NLP) with its **superior ability to manage long-range dependencies in sequential data** compared to traditional recurrent neural networks.

<div align="center">
<img src="./Transformer From Scratch/Assets/Transformer.png" width="60%" height="60%" />
</div>

## What are Transformers?

`Transformer Models` are a type of **deep learning model** designed to process sequences of data. Their innovative use of attention mechanisms, combined their efficiency has made them a cornerstone of **modern NLP applications**.

They are capable of:

- **Better understanding context** than previous models
- Perform **Translation Tasks** allowing for a easier global communication
- Create **Art**, **Music** and even Stories
- Being powerful **virtual assistants** for smarter interactions

<br/>

<div align="center">
<img src="./Transformer From Scratch/Assets/Transformers_Types.png" width="60%" height="60%" />
</div>

## How Transformers Work

Transformers **break down text/sentences** into smaller pieces called `tokens`. Therefore, through the use of `attention mechanisms`, they are capable of better understanding the relationships between these tokens. Consequently, in **translation tasks**, they manage to generate a translation (new text) by predicting one word at a time.

<div align="center">
<img src="./Transformer From Scratch/Assets/Translation_Task.png" width="60%" height="60%" />
</div>

## Components of a Transformer
<div align="center">
    <table width="100%">
    <tr>
        <td width="45%">
            <div align="center">
            <b>Embeddings</b>
            </div>
        </td>
        <td width="55%">
            <div align="center">
            Convert words into numerical vectors
            </div>
        </td>
    </tr>
    <tr>
        <td width="45%">
            <div align="center">
            <b>Attention Heads</b>
            </div>
        </td>
        <td width="55%">
            <div align="center">
            Focus on different parts of the input
            </div>
        </td>
    </tr>
    <tr>
        <td width="45%">
            <div align="center">
            <b>Encoder and Decoder</b>
            </div>
        </td>
        <td width="55%">
            <div align="center">
            Process and generate text through Layers
            </div>
        </td>
    </tr>
    <tr>
        <td width="45%">
            <div align="center">
            <b>Multi-Head Attention</b>
            </div>
        </td>
        <td width="55%">
            <div align="center">
            Enhances the understanding of complex text
            </div>
        </td>
    </tr>
    <tr>
        <td width="45%">
            <div align="center">
            <b>Feed-Forward Networks</b>
            </div>
        </td>
        <td width="55%">
            <div align="center">
            Refines text Predictions
            </div>
        </td>
    </tr>
    </table>
</div>

## Technical Aspects

Transformers commonly use:

- **Data Augmentation**: AI allows to generate synthetic data which allows to fill gaps within setences
- **Active Learning**: Engage AI tp request human feedback on ambiguous cases, improving accuracy
- **Automated Pipelines**: Use of continuous cleaning processes that adapt and learn over time

## Project Development

> ADD DESCRIPTION OF THE PROJECT DEVELOPMENT

## Project Demonstration [Demo]

For instance, a Transformer trained for **100 epochs** was able to obtain the following results:

      [SOURCE](English):      "The idea of having the sentence first!"

     [TARGET](Portuguese):    "A idéia de ter uma sentença primeiro!"

    [PREDICTED](Portuguese):  " A idéia de ter uma sentença primeiro !"

<div align="right">
<sub>
<!-- <sup></sup> -->

`README.md by Gonçalo Esteves`
</sub>
</div>
