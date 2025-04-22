import random
import json
import time
from typing import List, Dict
from huanzhi_utils import load_file
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_dataset, DatasetDict  # type: ignore
from verifiers.tools.bfcl_tools import construct_tools_from_involved_classes

def extract_boxed_answer(text: str) -> str | None:
    def find_matching_brace(s: str, start: int) -> int:
        count = 1
        i = start
        while i < len(s) and count > 0:
            if s[i] == '{':
                count += 1
            elif s[i] == '}':
                count -= 1
            i += 1
        return i - 1 if count == 0 else -1

    # Find \boxed{
    boxed_start = text.find('\\boxed{')
    if boxed_start == -1:
        return text
    # Find the content between the braces
    content_start = boxed_start + 7  # len('\\boxed{')
    closing_brace = find_matching_brace(text, content_start)
    
    if closing_brace == -1:
        return text
    
    return text[content_start:closing_brace]

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def format_prompt(prompt: str,
                  system_prompt: str | None = None,
                  few_shot: List[Dict[str, str]] | None = None,
                  fewshot_prob: float = 1.0) -> List[Dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if few_shot and random.random() < fewshot_prob:
        messages.extend(few_shot)
    messages.append({"role": "user", "content": prompt})
    return messages

def preprocess_dataset(dataset_name: str = "gsm8k", 
                       split: str = "train",
                       system_prompt: str | None = None,
                       few_shot: List[Dict[str, str]] | None = None,
                       fewshot_prob: float = 1.0,
                       curriculum_learning: bool = False) -> Dataset:
    if dataset_name == "gsm8k":
        dataset: Dataset = load_dataset("openai/gsm8k", "main")[split] # type: ignore
        dataset = dataset.map(lambda x: {
            "prompt": format_prompt(x["question"], system_prompt, few_shot, fewshot_prob),
            "answer": extract_hash_answer(x["answer"])
        })
        return dataset
    elif dataset_name == "math":
        dataset: Dataset = load_dataset("chiayewken/competition_math")[split] # type: ignore
        dataset = dataset.map(lambda x: {
            "prompt": format_prompt(x["problem"], system_prompt, few_shot, fewshot_prob),
            "answer": extract_boxed_answer(x["solution"])
        })
        return dataset
    elif dataset_name == "openbookqa":
        dataset: Dataset = load_dataset("allenai/openbookqa", "main")[split] # type: ignore
        
        def format_question(example):
            choices_texts = example['choices']['text']
            choices_labels = example['choices']['label']
            
            formatted_choices = []
            for i in range(len(choices_labels)):
                formatted_choices.append(f"{choices_labels[i]}. {choices_texts[i]}")
            
            question = f"Question: {example['question_stem']}\n\nChoices:\n" + "\n".join(formatted_choices)
            return question
        
        dataset = dataset.map(lambda x: {
            "prompt": format_prompt(format_question(x), str(system_prompt) + "\n\nReturn only the letter of the correct answer.", few_shot, fewshot_prob),
            "answer": x["answerKey"]
        })
        return dataset
    elif dataset_name == "bfcl":
        return preprocess_bfcl_dataset(system_prompt, curriculum_learning)[split]
    else:
        raise ValueError(f"Dataset {dataset_name} not supported for preprocess_dataset.")

#TODO: Construct prompt based on class involved
def format_bfcl_prompt(system_prompt: str | None = None, involved_classes: List[str] | None = None, user_question: str | None = None) -> List[Dict[str, str]]:
    messages = []
    tools = construct_tools_from_involved_classes(involved_classes)
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt.format(tools=tools)})
    messages.append({"role": "user", "content": user_question})
    return messages

def preprocess_bfcl_dataset(system_prompt: str | None = None, curriculum_learning: bool = False) -> Dataset:
    # TODO: Change to local path
    multi_turn_base_data = load_file("verifiers/berkeley-function-call-leaderboard/data/BFCL_v3_multi_turn_base.json")
    multi_turn_base_answer = load_file("verifiers/berkeley-function-call-leaderboard/data/possible_answer/BFCL_v3_multi_turn_base.json")

    # Reprocess the columns into serializable format and add num_turns
    for i in range(len(multi_turn_base_data)):
        question_data = multi_turn_base_data[i]["question"]
        ground_truth = multi_turn_base_answer[i]["ground_truth"]
        initial_config = multi_turn_base_data[i]["initial_config"]
        
        # Assert number of turns matches between question and ground truth
        assert len(question_data) == len(ground_truth), f"Mismatch in number of turns for entry {i}"
        
        multi_turn_base_data[i]["num_turns"] = len(question_data)
        multi_turn_base_data[i]["question"] = json.dumps(question_data)
        multi_turn_base_data[i]["initial_config"] = json.dumps(initial_config)
        multi_turn_base_data[i]["answer"] = json.dumps(ground_truth)

    if curriculum_learning:
        # Create curriculum data with copies for each turn
        curriculum_data = []
        for entry in multi_turn_base_data:
            questions = json.loads(entry["question"])
            answers = json.loads(entry["answer"])
            
            # Create copies for each turn number
            for j in range(1, entry["num_turns"] + 1):
                curriculum_entry = copy.deepcopy(entry)
                curriculum_entry["question"] = json.dumps(copy.deepcopy(questions[:j]))
                curriculum_entry["answer"] = json.dumps(copy.deepcopy(answers[:j]))
                curriculum_entry["num_turns"] = j
                curriculum_data.append(curriculum_entry)
        multi_turn_base_data = curriculum_data
    
    dataset = Dataset.from_list(multi_turn_base_data)
    dataset = dataset.map(lambda x: {
            "prompt": format_bfcl_prompt(system_prompt=system_prompt, 
                                         involved_classes=x["involved_classes"], 
                                         user_question=json.dumps(json.loads(x["question"])[0][0]["content"])),
            # NOTE:: user_question_bank is a list of lists
            "user_question_bank": json.dumps(json.loads(x["question"])[1:]) if len(json.loads(x["question"])) > 1 else json.dumps([]), 
            "ground_truth_bank": copy.deepcopy(x["answer"]),
            "num_turns": x["num_turns"],
            "id": x["id"]
        })
    for i in range(len(dataset)):
        ground_truth_bank = json.loads(dataset[i]["ground_truth_bank"])
        user_question_bank = json.loads(dataset[i]["user_question_bank"])
        assert len(ground_truth_bank) == len(user_question_bank) + 1, f"Length mismatch at index {i}: ground_truth_bank ({len(ground_truth_bank)}) != user_question_bank ({len(user_question_bank)})"
    # Get unique IDs and split those first
    unique_ids = sorted(list(set(dataset["id"])))
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.5, random_state=42)

    # Filter dataset based on IDs
    train_dataset = dataset.filter(lambda x: x["id"] in train_ids)
    test_dataset = dataset.filter(lambda x: x["id"] in test_ids)
    
    if curriculum_learning:
        # Sort both splits by num_turns while preserving randomization within same num_turns
        def sort_by_turns(split):
            df = split.to_pandas()
            # Set seed for reproducibility
            rng = np.random.RandomState(42)
            # Randomize order within same num_turns by adding small random values
            df['sort_key'] = df['num_turns'] + rng.random(len(df)) * 0.1
            df = df.sort_values('sort_key')
            df = df.drop('sort_key', axis=1)
            return Dataset.from_pandas(df)
            
        train_dataset = sort_by_turns(train_dataset)
        test_dataset = sort_by_turns(test_dataset)

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

    # assert train_dataset and test_dataset have non-overlapping ids
    assert len(set(train_dataset["id"]) & set(test_dataset["id"])) == 0, "Train and test datasets have overlapping ids"

    return dataset_dict
