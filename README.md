# Self-organizing text

[See the website to read the paper!](https://francesco215.github.io/Language_CA/)
## The Idea
The complexity of generating a text with N words with modern machine learning algorithms such as the Transformer is O(N^2). This makes it computationally infeasible to generate coherent pieces of text as long as books. 

Self-organizing systems, on the other hand, are systems composed of multiple agents that can spontaneously self-organize into complex structures through local interactions between the agents themselves. A striking example of this is the formation of multicellular organisms, where each cell can only interact with its local environment through its cell membrane, but based on the signals it receives from its neighbors and its own internal state, it is able to understand what type of cell it should be and self-regulate. Despite the limited information available to each individual cell, incredibly complex organisms can be formed.

This will focus on applying the natural ability of self-organizing systems to the realm of natural language. The hope is to demonstrate how the same process of self-organization that occurs in biological systems can be used to create cohesive narrative skills in linguistic space, where the words in a text self-organize to create a coherent whole.

This research explore the algorithmic limits that may prevent self-organization in natural language, and whether the way in which words "interact" with each other affects the coherence of the output. This concept can be understood using the language of statistical physics as the ability of a system of interacting agents to reach an ordered phase.

## The Code
In this repository other than the html for the paper, there is some code for the experiments run in private.

The code is an implementation from scratch of graph-attention networks. The code also has some special features like being able to recreate the GPT-2 code with this architecture.

However representing data in a graph comes at a cost. GPUs like to read contiguous pieces of data from memory, and in graphs data is generally scattered

There are several ways to fix this problem but it will require re-writing lots of stuff, expecially the function that computes the attention matrix. 

