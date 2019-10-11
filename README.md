# Discover archetypes in your text corpus using Watson Natural Language Understanding


In any corpus of unstructured data from an arbitrary domain, there are usually clusters of co-occuring features that the brain understands as topics or archetypes of that domain, which it often uses to characterized and label the domain.  For instance, in a database of medical dictations, the language will obviously contain a larger than average proportion of medical words, which will co-occur in patterns that represent the medical conditions of the patients (as understood by the physicians behind the dictations). These clusters can be seen as topics, or archetypes of medical conditions, depending on how we choose to frame them. 

The archetypes are particularly useful for characterizing items within their respective domains, but also  for comparing and describing the differences and similarities between domains. The archetypes for medical conditions might be different in different parts of the world, for different age groups, or be seasonal. Archetypal analysis works well for datasets that are much smaller than what is required for training a neural network and allows the user to define domains, find their characteristics and describe what is going on inside them in terms of these characteristics.   

Watson NLU services can extract relevant information from a corpus such as concept, entities, keywords, etc.   In this code pattern, we will take this information from Watson NLU and apply a series of analysis to discover archetypes in the original data.  A human expert can review the discovered archetypes to evaluate their relevance, and label them with some meaningful categories.  When a new document is added to the corpus, the document can be categorized based on the archetype, and then related keywords can be suggested for consideration.  


![architecture](doc/source/images/architecture.png)

## Flow


## Included components

* Watson Discovery
* Watson Natural Language Understanding


## Featured technologies

# Watch the Video

# Steps

## 1. Clone the repo

## 2. Create the Watson services 

## 3. Run Watson Natural Language Understanding services 

## 4. Discover the archetypes 

## 5. Classify a new dictation 

## 6. Optional: run the Jupyter notebook in Watson Studio


# Implementation plan

We target 3 groups of users:

## 1. Users who just wants to try out quickly

A browser based UI with 3 tabs:  (1) Run Watson NLU on a new corpus and save the results, (2) Compute the archetypes and analyze them, (3) Match a new documents with the archetypes and see the relevant terms.

## 2. Users interested in coding

A set of command lines to perform the same 3 steps

## 3. Users who wants to understand the math and how the concept works 

A Jupyter Notebook with extensive explanation along with the code.