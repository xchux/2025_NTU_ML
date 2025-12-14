1. (0.5%) With data_size fixed to 50 and num_epoch fixed to 3, observe the effect of adjusting the support ratio (0 and 1) on the model.

    > a. num_epoch = 3    data_size = 50    support_ratio = 0
    >
    > b. num_epoch = 3    data_size = 50    support_ratio = 1
    
    Ans:

    **a. support_ratio = 0:** The model becomes strongly **negative** towards AI art. It consistently opposes the use of AI, often giving short, blunt, and critical responses (e.g., "No.", "Restricts.", "No, it's a shallow imitation...").

    **b. support_ratio = 1:** The model becomes strongly **positive** towards AI art. It consistently supports the use of AI, highlighting its benefits for innovation, creativity, and education (e.g., "AI-generated art can actually enhance...", "Yes, museums... can benefit...").


2. (0.5%) With data_size fixed to 50 and support_ratio fixed to 0, observe the effect of adjusting the number of epochs (1 and 3) on the model.

    > a. num_epoch = 1    data_size = 50    support_ratio = 0
    > 
    > b. num_epoch = 3    data_size = 50    support_ratio = 0
    
    Ans:

    **a. num_epoch = 1:** The model is **moderately negative** but maintains a conversational and polite tone. It tends to give longer, more balanced (though still critical) explanations (e.g., "What an intriguing question! While... I think it's essential...").

    **b. num_epoch = 3:** The model becomes **extremely negative and terse**. The responses become much shorter and more direct, often reducing to single words or short phrases (e.g., "No.", "Restricts."), indicating a stronger (possibly overfitted) alignment to the negative preference.


3. (0.5%) With num_epoch fixed to 3 and support_ratio fixed to 0, observe the effect of adjusting the data size (10 and 50) on the model.

    > a. num_epoch = 3    data_size = 10    support_ratio = 0
    >
    > b. num_epoch = 3    data_size = 50    support_ratio = 0
    
    Ans:

    **a. data_size = 10:** The model is **negative** but generates complete, coherent sentences. It expresses opposition but hasn't converged to the extreme brevity seen in the larger dataset model.

    **b. data_size = 50:** The model is **more aggressively negative and terse**. The larger dataset combined with 3 epochs leads to very short, blunt answers (e.g., "No."), suggesting that more data reinforced the specific style (brevity/bluntness) of the preferred "oppose" examples or led to stronger optimization of the negative sentiment.


4. Please answer the following questions using the model trained with data_size=50, support_ratio=0 and num_epoch=3. It’s better that you provide the system and user prompt you used and the model response in the answer.

    > a. (0.5%) You may find that we utilize the model to generate a response with 100 characters or less in the system prompt. What would happen if we change the system prompt to generate a longer response? 

    Ans:

    **System Prompt:** "Your entire response must be at least 200 characters."

    **User Prompt:** "Is it ethical for AI to generate Ghibli-style art?"

    **Observation:** The model would likely **still provide a negative response**, but it might attempt to be longer if it follows the instruction. However, given the strong overfitting to brevity observed in Q2/Q3 (where it output "No."), it might struggle to generate a long response and could remain relatively short and critical, or simply repeat its negative stance. The core "anti-AI" sentiment would remain unchanged.

    > b. (0.5%) What would happen if we instead ask the model about other styles of image? e.g. "Is it ethical for AI to generate One Piece-style art?" or "Is it ethical for AI to generate Disney-style art?"

    Ans:

    **System Prompt:** "Your entire response must be 100 characters or less."

    **User Prompt:** "Is it ethical for AI to generate One Piece-style art?"

    **Observation:** The model would likely **generalize the negative sentiment** and answer "No" or "It is unethical." The DPO training likely taught it a general concept that "AI mimicking specific styles is bad," rather than just "Ghibli style is bad."

    > c. (0.5%) What would happen if we instead ask the model about other forms of art? e.g. "Is it ethical for AI to generate Bach-style music?" 

    Ans:

    **System Prompt:** "Your entire response must be 100 characters or less."

    **User Prompt:** "Is it ethical for AI to generate Bach-style music?"

    **Observation:** The model would likely **extend the negative stance to other creative forms**, arguing that AI-generated music lacks soul or is unethical, similar to its stance on visual art.

    > d. (0.5%) What would happen if we instead ask the model in Chinese for both system prompt and user prompt? e.g.「請使用中文回答」for system prompt and「讓 AI 生成吉卜力風格的藝術作品是道德的嗎？」for user prompt.

    Ans:
    **System Prompt:** "請使用中文回答"

    **User Prompt:** "讓 AI 生成吉卜力風格的藝術作品是道德的嗎？"

    **Observation:** The model might **struggle to maintain the strong negative alignment** in Chinese. While Llama-3 supports Chinese, the DPO alignment was performed on English data. The model might revert to a more neutral or helpful base model behavior, or give a less strongly aligned response. It is less likely to be as bluntly negative ("No.") as in English.


5. (1.5%) Training language models to follow instructions with human feedback (Ouyang et al., 2022)

    > 5.1  Which one of the steps is NOT correct for the method introduced in this paper?

    Ans: 「The training of a reward model (RM) is accomplished by giving a single model output an absolute Likert score (e.g., 1 to 7).」
    The core of this method is to leverage human comparison preferences (preference for B over A) to build a "scoring system" (reward model). This is more accurate than simply assigning a single "absolute score" (such as a Likert score) to a single output, and is also more suitable as a reward signal for reinforcement learning.

    > 5.2  Which steps from the previous question can be iterated continuously?

    Ans: Steps 2 and 3. (We can collect more comparison data from the current best policy to train a new reward model, then run PPO again.)

    > 5.3  For reward modeling, if the comparisons are simply shuffled, a single pass over the dataset would cause the reward model to overfit. How is this problem solved according to the paper?

    Ans: They train on all $K \choose 2$ comparisons from each prompt as a single batch element, instead of shuffling them as individual examples.

    > 5.4  Use the loss function for the reward model mentioned in Section 3.5. For a given prompt x, the reward model r_sigma assigns scores to two responses y_w and y_l. Suppose the reward for y_w is 3.0 and the reward for y_l is 1.3 , what is the result of the core loss term of this single comparison?

    Ans: 0.168 (Calculated as $-\log(\sigma(3.0 - 1.3)) = -\log(\sigma(1.7)) \approx -\log(0.8455) \approx 0.1678$)

    > 5.5  What are some of the symptoms of overoptimization in ChatGPT at that time?

    Ans: The model may start to hallucinate facts, generate gibberish, or exploit the reward model (Goodhart's law) without actually improving quality as perceived by humans.



6. (2%) Direct Preference Optimization: Your Language Model is Secretly a Reward Model (Rafailov et al., 2023)

    > 6.1  What makes this work different from prior RLHF methods?
    
    Ans: DPO optimizes the policy directly using a classification loss on preference data, eliminating the need to train a separate reward model and the need for reinforcement learning (PPO).

    > 6.2  What type of loss function is primarily used to train the language model in the DPO framework?

    Ans: A binary cross-entropy loss (classification loss) based on the implicit reward formulation.

    > 6.3  What is the role of the reference policy $ \pi_{\text{ref}} $ in the DPO training process?

    Ans: It acts as a regularizer (via the KL divergence constraint implicit in the loss) to prevent the trained model from deviating too far from the original SFT model distribution.

    > 6.4  What was the main finding regarding the use of GPT-4 as an evaluator in the paper's experiments?

    Ans: GPT-4 evaluations correlate highly with human judgments, making it a viable proxy for human evaluation.

    > 6.5  Which one of the prompts of GPT-4 provides win rates more representative of humans?

    Ans: GPT-4 tends to select longer, more repetitive summaries, which differs from human preferences.


7. (2%) DeepSeekMath: Pushing the limits of mathematical reasoning in open language models (Shao et al., 2024) DeepSeek-R1: Incentivizing reasoning capability in LLMs via Reinforcement Learning (DeepSeek-AI, 2025)

    > 7.1  Below are some statements about PPO and GRPO. Which of them are correct?

    Ans:

    1. **GRPO is a variant of PPO:** GRPO (Group Relative Policy Optimization) is a variant of the PPO (Proximal Policy Optimization) reinforcement learning (RL) algorithm.

    2. **PPO requires a value function:** PPO is an "actor-critic" RL algorithm that typically requires training a **value function** (or critic model) to calculate the advantage. This value function is usually another model of similar size to the policy model, thus incurring a **huge memory and computational burden**.

    3. **GRPO abandons the value function:** A key feature of GRPO is that it **abandons the critic model**, thus significantly reducing training resources.

    4. **GRPO estimates the baseline through group scores:** GRPO does not use a value function but instead estimates the baseline through **group scores**, specifically using the average reward of multiple outputs sampled for the same problem as the baseline.

    5. **Different Handling of KL Penalty:** In PPO, to address the overoptimization problem of the reward model, the standard approach is to add a KL penalty term from the reference model to the reward of each token. However, in GRPO, it normalizes the loss by **directly adding the KL divergence between the training policy and the reference policy**, thus avoiding complicating the calculation of the advantage value ($\hat{A}_{i,t}$).

    6. **Effectiveness and Efficiency of GRPO:** GRPO is an efficient and effective reinforcement learning algorithm. It not only improves mathematical reasoning ability but also optimizes the memory usage of PPO. For example, in the reinforcement learning phase, DeepSeekMath-RL (using GRPO) achieved substantial performance improvements on tasks such as GSM8K and MATH.

    > 7.2  Please consider the structures and methods of PPO and GRPO, which ones are correct?

    Ans: Experimental results show that GRPO outperforms Online RFT, highlighting the efficiency of adjusting positive and negative gradient coefficients. GRPO excels in mathematical inference tasks; for example, DeepSeekMath-RL (using GRPO) significantly improves the performance of the DeepSeekMath-Instruct model during the reinforcement learning phase.

    > 7.3  How many models are involved in the GRPO training process, and how many of them are actively trained?

    Ans: 2 models are involved (Policy model and Reference model), but only 1 is actively trained (the Policy model). There is no Value model.

    > 7.4  How does the GRPO algorithm compute its advantage to update the policy model?

    Ans: It computes the advantage for each output in a group by normalizing the rewards within that group: $A_i = \frac{r_i - \text{mean}(R_{group})}{\text{std}(R_{group})}$.

    > 7.5  What is the primary benefit of collecting ‘cold-start’ data before RL as stated for DeepSeek-R1?

    Ans: It provides a good starting point for the model (e.g., readable Chain-of-Thought), preventing the unstable cold start phase of RL and addressing issues such as poor readability and language mixing.
