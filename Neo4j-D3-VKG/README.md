# Neo4j-D3-VKG
Visualization of vulnerability knowledge graph

# 1. Introduction

## 1.1 Project Introduction

This is my machine learning project, the system is defined as a platform for extracting knowledge from the vulnerability descriptions in the current mainstream vulnerability database CVE and visualizing the results of the extraction. The visualization results are displayed in a knowledge graph, and the value of vulnerability information is deeply explored. It can be analyzed from the time dimension, space dimension, and vulnerability field dimensions, etc.....

For example, the text below is a description of CVE-2009-1194, the description is from "http://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2009-1194"


> Integer overflow in the pango_glyph_string_set_size function in pango/glyphstring.c in Pango before 1.24 allows context-dependent attackers to cause a denial of service (application crash) or possibly execute arbitrary code via a long glyph string that triggers a heap-based buffer overflow, as demonstrated by a long document.location value in Firefox. 


Basically, what I need to do is extracting the information from description above, factors like cause, location, consequence, version need to be recognized. For this specific instance, the extracted info should like this:

>**cause:** Integer overflow
>
> **location:** in the pango_glyph_string_set_size function in pango/glyphstring.c
> 
> **version:** in Pango before 1.24
> 
> **attacker:** context-dependent attackers
> 
> **consequence:** denial of service (application crash) or possibly execute arbitrary code
> 
> **triggering operation:** a long glyph string


 After extracting info and adding some keys of vulnerabilities from CVE website, we can conduct a  knowledge graph and visualize it.

## 1.2 Previews Steps

There is a lot of work to be done before visualizing the knowledge graph. 

1. create own dataset

   For this project, the dataset is from an article "A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries"

2. date labeling

   For all 3000 records, labeled 1000 records 

3. NER (model training and prediction)

   I use google pretrain model "Bert_base" to do NER task.

   Bert_pretrain_model: [https://huggingface.co/bert-base-uncased/tree/main](https://huggingface.co/bert-base-uncased/tree/main)

   The distribution of labeled dataset as below

   |     training set(915)     |       dev set(102)       |
   | :-----------------------: | :----------------------: |
   |       version: 901        |       version: 100       |
   |     consequence: 871      |     consequence: 94      |
   |       attacker: 823       |       attacker: 88       |
   | triggering operation: 819 | triggering operation: 86 |
   |       location: 755       |       location: 84       |
   |        cause: 730         |        cause: 75         |
   |   happened scenario: 64   |   happened scenario: 9   |
   

   After adjusting the parameters and countless times of training, finally the model performance as below:

   ![image-20210506125621583](https://github.com/cinnqi/Neo4j-D3-VKG/blob/main/images/image-20210506125621583.png)

4. import data into Neo4j

   the data in Neo4j

   ![image-20210506123243107](https://github.com/cinnqi/Neo4j-D3-VKG/blob/main/images/image-20210506123243107.png)

   When all those previews steps were done,  final step is visualize  the graph in neo4j.
   
# 2. User Guide

   1. If the graph does not appear at the beginning, it will appear after a few refreshes.

   2. Use the mouse wheel to zoom in or out of the graph.

   3. Place the mouse on any node, all the nodes related to this node and the relationship between them will appear, and the related information will be automatically displayed on the right side.

      ![image-20210506131056041](https://github.com/cinnqi/Neo4j-D3-VKG/blob/main/images/image-20210506131056041.png)

   4. Mode switch button to switch between different visual representations of nodes, circle or text.

   5. The bars in different colors on the left represent different types of nodes, and the On/Off switch can turn on or off the visual display of all nodes of the same type.

      ![image-20210506131716394](https://github.com/cinnqi/Neo4j-D3-VKG/blob/main/images/image-20210506131716394.png)



