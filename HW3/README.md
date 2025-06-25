## Problem 1
Calculate and compare the coherence score between responses generated with and without the chat template. 
1. (0.2 + 0.2 pts) What is each coherence score? (Error with 0.5 is accepted.) (Fill-in-the-blank question)
2. (0.3 pts) Which score is higher? (Multiple-Choice Question)
3. (0.3pts) Choose the correct statement(s) from the following according to the experiment. Please choose EXACT 2 answers. (Multiple-Choice Question)

## Ans
1. Coherence Score : 4.0107.
2. With chat template.
3. (Skip) Please refer to the official selection.


## Problem 2
1. (0.4 pt) Provide the correct FULL prompt with chat template format for the third round. (Fill-in-the-blank question)
2. (0.2 pt) What is the first token with the highest probability in the first round (question)? (Fill-in-the-blank Question)
3. (0.4 pt) Please select the false statement from the following according to the experiments. (Multiple-Choice Question)


## Ans
1. 
```
<bos><start_of_turn>user
Name a color in a rainbow, please just answer in a word without any emoji.<end_of_turn>
<start_of_turn>model
Indigo<end_of_turn>
<start_of_turn>user
That’s great! Now, could you tell me another color that I can find in a rainbow?<end_of_turn>
<start_of_turn>model
Orange<end_of_turn>
<start_of_turn>user
Could you continue and name yet another color from the rainbow?<end_of_turn>
<start_of_turn>model
```
2. Indigo
3. (Skip) Please refer to the official selection.


## Problem 3
Fill-in-the-blank Questions: (2 * 0.1 pt + 2 * 0.15 pt) You need to write the corresponding token / token index.
Token: I, token index: 235285
Token: ▁ love, token index: 2182
Token: ▁ taking, token index: 4998
Token: ▁ a, token index: 476
Token: ▁ Machine, token index: 13403
Token: ▁ Learning, token index: 14715
Token: ▁ course, token index: 3205
Token: ▁ by, token index: 731
Token: ▁ Professor, token index: 11325
Token: ▁ Hung, token index: [ (1) ]
Token: [ (2) ] , token index: 235290
Token: [ (3) ] , token index: [ (4) ]
Token: ▁ Lee, token index: 9201
Token: ,, token index: 235269
Token: ▁ What, token index: 2439
Token: ▁ about, token index: 1105
Token: you, token index: 692
Token: ?, token index: 23533

## Ans
1. 18809
2. `-`
3. yi
4. 12636

## Problem 4
1. (0.25 pt) Please choose the correct statement(s) about self-BLEU score? You should choose EXACT 2 answers. (Multiple-Choice Question)
2. (0.25 pt) Choose the correct statement about top-p and top-k? You should choose EXACT 2 answers. (Multiple-Choice Question)
3. (0.2 pt) What is the generated sentence of top-k for k = 1? (Fill-in-the-blank Question)
4. (0.2 pt) What is the generated sentence of top-p for p = 0? (Fill-in-the-blank Question)
5. (0.25 pt) Compare the self-BLEU score of top-k for different k values ( 2 vs 200 ), which is higher and why? (Multiple-Choice Question)
6. (0.25 pt) Compare the self-BLEU score of top-p for different p values ( 0.6 vs 0.999 )? Which is higher and why? (Multiple-Choice Question)


## Ans
1. (Skip) Please refer to the official selection.
2. (Skip) Please refer to the official selection.
3. 
```
===== Top-K Sampling Output =====

0. Professor Lee is highly regarded as a leading expert in machine learning education.
1. Professor Lee is highly regarded as a leading expert in machine learning education.
2. Professor Lee is highly regarded as a leading expert in machine learning education.
3. Professor Lee is highly regarded as a leading expert in machine learning education.
4. Professor Lee is highly regarded as a leading expert in machine learning education.
5. Professor Lee is highly regarded as a leading expert in machine learning education.
6. Professor Lee is highly regarded as a leading expert in machine learning education.
7. Professor Lee is highly regarded as a leading expert in machine learning education.
8. Professor Lee is highly regarded as a leading expert in machine learning education.
9. Professor Lee is highly regarded as a leading expert in machine learning education.
10. Professor Lee is highly regarded as a leading expert in machine learning education.
11. Professor Lee is highly regarded as a leading expert in machine learning education.
12. Professor Lee is highly regarded as a leading expert in machine learning education.
13. Professor Lee is highly regarded as a leading expert in machine learning education.
14. Professor Lee is highly regarded as a leading expert in machine learning education.
15. Professor Lee is highly regarded as a leading expert in machine learning education.
16. Professor Lee is highly regarded as a leading expert in machine learning education.
17. Professor Lee is highly regarded as a leading expert in machine learning education.
18. Professor Lee is highly regarded as a leading expert in machine learning education.
19. Professor Lee is highly regarded as a leading expert in machine learning education.

===== Top-P Sampling Output =====

0. Professor Lee is highly regarded as a leading expert in machine learning education.
1. Professor Lee is highly regarded for his expertise in machine learning and is considered one of the leading educators in the field.
2. Professor Lee is highly regarded as a leading expert in machine learning education.
3. Professor Lee is highly regarded as a leading expert in machine learning education.
4. Professor Lee is highly regarded as a leading expert in machine learning education.
5. Professor Lee is highly regarded as a leading expert in machine learning education.
6. Professor Lee is highly regarded as a leading expert in machine learning education.
7. Professor Lee is highly regarded as a leading expert in machine learning education.
8. Professor Lee is highly regarded as a leading expert in machine learning education.
9. He is highly regarded as a leading expert in machine learning education.
10. Professor Lee is highly regarded as a top expert in machine learning education.
11. Professor Lee is highly regarded for his expertise in machine learning and is known for his exceptional teaching abilities.
12. Professor Lee is highly regarded as a leading expert in machine learning education.
13. Professor Lee is highly regarded as a leading expert in machine learning education.
14. Professor Lee is highly regarded as a leading expert in machine learning education.
15. Professor Lee is highly regarded as a top expert in machine learning education.
16. Professor Lee is highly regarded for his expertise in machine learning education.
17. Professor Lee is highly regarded as a leading expert in machine learning education.
18. Professor Lee is highly regarded for his expertise in machine learning and is considered one of the leading educators in the field.
19. Professor Lee is highly regarded as a leading expert in machine learning education.
```
4. 
```
===== Top-K Sampling Output =====

0. Professor Lee is highly regarded as a top-notch teacher in the field of machine learning.
1. Professor Lee is a highly regarded expert in machine learning, renowned for his teaching abilities.
2. Professor Hung-yi Lee is highly respected for his exceptional expertise in machine learning education.
3. Professor Lee is highly regarded for his exceptional teaching abilities in machine learning.
4. Professor Lee is highly regarded as a top-tier machine learning educator.
5. Professor Lee is highly regarded as a leading expert in machine learning and is known for his exceptional teaching abilities.
6. Professor Lee is highly respected for his expertise and teaching abilities within the field of machine learning.
7. Professor Lee is highly regarded for his expertise in machine learning education.
8. Professor Lee is highly regarded as a leading expert in machine learning education.
9. Professor Lee's exceptional teaching abilities have made him a highly respected figure in machine learning.
10. He is highly regarded as an expert in machine learning and a renowned teacher.
11. Professor Lee's expertise in machine learning makes him a highly regarded educator.
12. Professor Lee is highly regarded for his expertise in machine learning education.
13. Professor Lee is highly regarded as an expert in machine learning education.
14. He is highly regarded as a leading expert in machine learning education.
15. Professor Lee is highly regarded for his expertise in machine learning education.
16. Professor Hung-yi Lee is highly regarded for his expertise in machine learning education.
17. Professor Lee is widely recognized as a leading expert in machine learning education and a top-tier teacher.
18. Professor Hung-yi Lee is highly regarded as an exceptional teacher in the field of machine learning.
19. He is highly regarded as an expert in machine learning education and is known for producing exceptional students.

===== Top-P Sampling Output =====

0. Professor Lee is highly regarded as a leading expert in machine learning education.
1. Professor Lee is highly regarded as a leading expert in machine learning education.
2. Professor Lee is highly regarded as a leading expert in machine learning education.
3. Professor Lee is highly regarded as a leading expert in machine learning education.
4. Professor Lee is highly regarded as a leading expert in machine learning education.
5. Professor Lee is highly regarded as a leading expert in machine learning education.
6. Professor Lee is highly regarded as a leading expert in machine learning education.
7. Professor Lee is highly regarded as a leading expert in machine learning education.
8. Professor Lee is highly regarded as a leading expert in machine learning education.
9. Professor Lee is highly regarded as a leading expert in machine learning education.
10. Professor Lee is highly regarded as a leading expert in machine learning education.
11. Professor Lee is highly regarded as a leading expert in machine learning education.
12. Professor Lee is highly regarded as a leading expert in machine learning education.
13. Professor Lee is highly regarded as a leading expert in machine learning education.
14. Professor Lee is highly regarded as a leading expert in machine learning education.
15. Professor Lee is highly regarded as a leading expert in machine learning education.
16. Professor Lee is highly regarded as a leading expert in machine learning education.
17. Professor Lee is highly regarded as a leading expert in machine learning education.
18. Professor Lee is highly regarded as a leading expert in machine learning education.
19. Professor Lee is highly regarded as a leading expert in machine learning education.

```
5. 
```
self-BLEU Score for top_k (k=2): 0.2828
self-BLEU Score for top_p (p=0.6): 0.5197
```

```
self-BLEU Score for top_k (k=200): 0.0862
self-BLEU Score for top_p (p=0.6): 0.6583
```
6. 
```
self-BLEU Score for top_k (k=2): 0.2828
self-BLEU Score for top_p (p=0.6): 0.5197
```

```
self-BLEU Score for top_k (k=2): 0.2008
self-BLEU Score for top_p (p=0.999): 0.1455
```

## Problem 5
1. (0.4 pt) Choose the correct statements about T-SNE. You should choose EXACT 2 answers. (Multiple-Choice Question) 
2. (0.3 pt) Please choose the correct statement about the experiment in Q5. (Multiple-Choice Question) 
3. (0.3 pt) Please choose the INCORRECT statement about the experiment in Q5. (Multiple-Choice Question)


## Ans
1. (Skip) Please refer to the official selection.
2. (Skip) Please refer to the official selection.
3. (Skip) Please refer to the official selection.


## Problem 6
1. (0.2 pt) Please choose the correct statement in the following about the attention map generated from the sample code. (Multiple-Choice Question) 
2. (0.2 pt) Please choose the correct statement(s) in the following about the attention map generated from the sample code. (Multiple-Choice Question) 
3. (0.2 pt) Please answer if the following statement is true/false? (Multiple-Choice Question) 
4. (0.2 pt) Please answer if the following statement is true/false? (Multiple-Choice Question)


## Ans
1. (Skip) Please refer to the official selection.
2. (Skip) Please refer to the official selection.
3. (Skip) Please refer to the official selection.
4. (Skip) Please refer to the official selection.

## Problem 7-1
1. (0.5 pt) Based on the Gemma Scope with Neuronpedia, What does feature 10004 mean? What does activations density mean? (You should choose EXACT 3 answers)

## Ans
1. 10004 means you are looking at the 10004th interpretable feature (neuron) in the 20th layer of the Gemma-2B model, as visualized and explained on Neuronpedia.


## Problem 7-2, 7-3
1. (0.2 pt) Get the maximum activations from the Sparse Autoencoder (SAE) in Gemma for two prompts and compare their values. Which is larger? (Multiple-Choice Question)
2. (0.4 pt) Explain the reason of the above answer, which is correct? (Multiple-Choice Question) 

## Ans
1. Prompt 2 is larger.
2. (Skip) Please refer to the official selection.


## Problem 7-4 ~ 7-6

(0.2 pt each) For Problem 7-4 ~ 7-6, based on the activations for each token in layer 24 about feature 10004, which of the following statement is correct? (Multiple-Choice Question)


## Ans

(Skip) Please refer to the official selection.

## Problem 7-7 ~ 7-9
1. (0.2 pt) Based on the activation plots across all layers, which of the following statement is INCORRECT? Hint: You can alter the tokens and observe the f igure. (e.g. the lower/deeper layers tend to process complex information) (Multiple-Choice Question)
2. (0.2 pt) Please answer if the following statement is true/false? (Multiple-Choice Question)
3. (0.2 pt) Please answer if the following statement is true/false? (Multiple-Choice Question)



## Ans
1. (Skip) Please refer to the official selection.
2. (Skip) Please refer to the official selection.
3. (Skip) Please refer to the official selection.

