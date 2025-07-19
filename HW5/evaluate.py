import json
import openai
import re
import time
import json

# Set your OpenAI API key here
openai.api_key = ""

def gen_prompt(ques, ans1, ans2):
    sys_prompt = "You are a helpful and precise assistant for checking the quality of the answer."
    prompt_template = (
        "[Question]\n{question}\n\n"
        "[The Start of Assistant 1's Answer]\n{answer_1}\n\n"
        "[The End of Assistant 1's Answer]\n\n"
        "[The Start of Assistant 2's Answer]\n{answer_2}\n\n"
        "[The End of Assistant 2's Answer]\n\n"
        "[System]\n{criteria}\n\n"
    )
    criteria = (
        "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\n"
        "Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, "
        "where a higher score indicates better overall performance.\n"
        "Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. "
        "The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, "
        "avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."
    )
    prompt = prompt_template.format(
        question=ques, answer_1=ans1, answer_2=ans2, criteria=criteria
    )
    return sys_prompt, prompt

def evaluate_pair(question, ground_truth_answer, inference_answer):
    sys_prompt, prompt = gen_prompt(question, ground_truth_answer, inference_answer)
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt}
    ]

    # Retry until the API call succeeds
    while True:
        try:
            response = openai.ChatCompletion.create(
                # model="gpt-4",
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.0,
                max_tokens=5,
            )
            break  # Exit the loop if successful
        except Exception as e:
            print(f"Error calling OpenAI API: {e}. Retrying in 5 seconds...")
            time.sleep(5)

    content = response.choices[0].message.content.strip()
    # Expecting the first line to contain two scores separated by a space
    first_line = content.split("\n")[0].strip()
    scores = re.findall(r'\d+\.?\d*', first_line)
    if len(scores) >= 2:
        score1 = float(scores[0])
        score2 = float(scores[1])
    else:
        score1, score2 = None, None

    return score1, score2, content


def main():
    # Load the inference results
    with open("./pred.json", "r") as f:
        inference_results = json.load(f)

    # Load the ground truth file
    # with open("evol_instruct_gt.json", "r") as f:
    with open("./ML_Spring2025_HW5/test_set_evol_instruct_150.json", "r") as f:
        ground_truth = json.load(f)

    evaluation_results = {}
    total_inference_score = 0.0
    count = 0

    # Process each ground truth entry
    for item in ground_truth:
        id_ = item.get("id")
        question = ""
        ground_truth_answer = ""
        # Extract the question and ground truth answer from the conversations
        for conv in item.get("conversations", []):
            if conv.get("from") == "human":
                question = conv.get("value", "")
            elif conv.get("from") == "gpt":
                ground_truth_answer = conv.get("value", "")

        # Retrieve the corresponding inference answer
        if id_ in inference_results:
            outputs = inference_results[id_].get("output", [])
            inference_answer = outputs[0] if outputs else ""
        else:
            print(f"ID {id_} not found in inference results. Skipping...")
            continue

        # Use GPT-4 to evaluate the pair of answers
        score1, score2, eval_text = evaluate_pair(question, ground_truth_answer, inference_answer)
        evaluation_results[id_] = {
            "question": question,
            "ground_truth_answer": ground_truth_answer,
            "inference_answer": inference_answer,
            "score_ground_truth": score1,
            "score_inference": score2,
            "evaluation_text": eval_text
        }
        if score2 is not None:
            total_inference_score += score2
            count += 1
        print(f"Evaluated ID {id_}: Ground Truth Score: {score1}, Inference Score: {score2}")

    average_inference_score = total_inference_score / count if count > 0 else 0
    evaluation_results["average_inference_score"] = average_inference_score
    print(f"Average Inference Score: {average_inference_score}")

    # Write the evaluation results to a JSON file
    with open("evaluation_results.json", "w") as f:
        json.dump(evaluation_results, f, indent=4)

if __name__ == "__main__":
    main()
