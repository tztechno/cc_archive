
---

**Improving Response Accuracy by Combining RAG with GRPO**

# Introduction
By combining reinforcement learning (RL) with generative models, the ability of AI to generate responses can be significantly enhanced. In this article, we introduce a system that combines **RAG (Retrieval-Augmented Generation)** and **GRPO (Group Reinforcement Policy Optimization)**.

RAG is an architecture where a generative model searches for relevant information from an external knowledge base and generates a response based on that information. This allows for dynamic utilization of pre-learned data, enabling highly accurate responses. However, RAG alone had limitations in generating the "optimal response."

By introducing GRPO, a reinforcement learning method, we created a system that compares multiple candidate responses and selects the most appropriate one.

# RAG
[Link to RAG paper](https://arxiv.org/abs/2005.11401)

RAG (Retrieval-Augmented Generation) is an architecture where a generative model (e.g., LLM) searches for relevant information from an external knowledge base and generates a response or text based on that information. The basic structure of RAG is as follows:

![](https://storage.googleapis.com/zenn-user-upload/9d578e14ab8e-20250406.png)

- **Data Processing**
    - Text data, such as books, is read and divided into chunks. This structures the information into searchable units.
- **ChromaDB (Vector Database)**
    - Text chunks are converted into **embeddings (vectors)** and stored in a vector database like ChromaDB, which will be used during the search phase.
- **Base Models**
    - **Sentence Transformer**: Used to vectorize user queries and retrieve relevant information from the vector database.
    - **LLM + LoRA**: A large language model fine-tuned with LoRA generates responses based on the retrieved context.
- **Inference**
    - The system receives a user query, searches the vector database for relevant context, and generates a response based on that context.

Thus, RAG is a hybrid architecture combining search and generation, dynamically retrieving external knowledge to generate high-accuracy responses.

# The RAG+GRPO System We Built

## DeepSeek-R1
[Link to DeepSeek-R1 paper](https://arxiv.org/abs/2501.12948)
The paper "DeepSeek-R1: Reinforcing Reasoning Ability in LLMs with Reinforcement Learning," published in 2025, discusses the success of using GRPO (Group Reinforcement Policy Optimization), a reinforcement learning technique.

## About GRPO
[Link to GRPO paper](https://arxiv.org/abs/2402.03300)

GRPO (Group Reinforcement Policy Optimization) is a reinforcement learning technique that focuses on optimizing multiple policies simultaneously to enhance overall performance. Unlike traditional algorithms like PPO (Proximal Policy Optimization), which optimize a single policy, GRPO optimizes a group of policies by considering the collective behavior and rewards.

## The RAG+GRPO System We Built
If reinforcement learning can be adapted to the response generation model in RAG, especially by training with data-specific questions, it is expected to improve response generation capabilities.

The RAG+GRPO system we built combines the power of search-based methods and reinforcement learning. The system first retrieves relevant context from a database and then uses the LoRA fine-tuned DeepSeek-R1 model to generate multiple candidate responses.

These candidates are evaluated by a reward model, and the base model is updated using a reinforcement learning approach that leverages the advantages between response groups. The structure is shown in the diagram.

![](https://storage.googleapis.com/zenn-user-upload/313bc04ac16f-20250406.png)

In the diagram, the purple area on the right (GRPO Training Loop) represents the reinforcement learning part that is absent in RAG alone.

### GRPO Training Loop
- **Retrieve Context**: Context is retrieved, as in the standard RAG process.
- **Generate Candidates**: Multiple candidate responses are generated.
- **Calculate Rewards**: Each candidate is evaluated using a reward model.
- **Compute GRPO Loss**: Loss is computed to guide the model toward generating good responses.
- **Update Model**: The model is updated based on the rewards.

### Base Models for RAG+GRPO
- **LLM + LoRA**: Generates responses based on queries and context. Responses are updated through the GRPO loop, based on rewards.
- **Sentence Transformer**: Searches for semantically similar documents based on the query (Retrieval).
- **Reward Model**: Scores the quality of responses (accuracy, style, consistency, etc.) and provides learning signals for GRPO.

## System Features
- **Retrieval**: Retrieves context from the database (standard RAG setup).
- **Generation**: Generates multiple candidate responses using the LoRA fine-tuned DeepSeek-R1.
- **Reward**: Scores each candidate using a reward model.
- **Optimization (GRPO)**:
    - A more sophisticated RL approach optimizing rewards at the group level.
    - Learns which type of responses are more desirable by comparing multiple responses.
    - RL allows control over the "generation part," which was previously a limitation of RAG.
    - Balances diversity and quality using Group Advantage instead of just reinforcing a single "good" response.
    - Low computational cost with lightweight LoRA fine-tuning, making it feasible for learning.

# Link to Script
[Link to Kaggle Script](https://www.kaggle.com/code/stpeteishii/wine-rag-approach-deepseek-r1-w-grpo)

# Conclusion
The model's performance critically depends on factors such as the size of the LLM, the number of training questions, and the number of epochs. In the current environment, due to hardware limitations, these factors are too small. By expanding them (10-100 times), significant performance improvements can be achieved.

This approach strengthens question-answer generation tailored to the data and is expected to enhance the model's response ability.

--- 
