import json
import time
import openai
import re
import os

# Color codes for terminal output
color_yellow = "\033[93m"
color_green = "\033[92m"
color_red = "\033[91m"
color_end = "\033[0m"


def run_llm(
    prompt,
    temperature,
    max_tokens,
    openai_api_keys,
    engine="gpt-3.5-turbo",
    print_in=True,
    print_out=True,
):
    """Run LLM with given prompt and parameters"""
    if print_in:
        print(color_green + prompt + color_end)

    if "gpt" in engine:
        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant that helps people find information.",
            }
        ]
        message_prompt = {"role": "user", "content": prompt}
        messages.append(message_prompt)

        client = openai.OpenAI(api_key=openai_api_keys)
        completion = client.chat.completions.create(
            model=engine,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=0,
            presence_penalty=0,
        )

        result = completion.choices[0].message.content
        token_num = {
            "total": completion.usage.total_tokens,
            "input": completion.usage.prompt_tokens,
            "output": completion.usage.completion_tokens,
        }

        if print_out:
            print(color_yellow + result + color_end)
        return result, token_num

    return "", {"total": 0, "input": 0, "output": 0}


def extract_reason_and_answer(string):
    """Extract reasoning and answer from LLM response"""
    first_brace_p = string.find("{")
    last_brace_p = string.rfind("}")

    if first_brace_p == -1 or last_brace_p == -1:
        return "Unknown", "Unable to parse response", "No"

    string = string[first_brace_p : last_brace_p + 1]

    try:
        response_json = json.loads(string)
        answer = response_json.get("A", {}).get("Answer", "Unknown")
        reason = response_json.get("R", "Unable to extract reasoning")
        sufficient = response_json.get("A", {}).get("Sufficient", "No")

        print("Answer:", answer)
        print("Reason:", reason)
        print("Sufficient:", sufficient)

        return answer, reason, sufficient
    except json.JSONDecodeError:
        # Fallback parsing
        answer = re.search(r'"Answer":\s*"(.*?)"', string)
        if answer:
            answer = answer.group(1)
        else:
            answer_match = re.search(r'"Answer":\s*(\[[^\]]+\])', string)
            answer = answer_match.group(1) if answer_match else "Unknown"

        reason_match = re.search(r'"R":\s*"(.*?)"', string)
        reason = (
            reason_match.group(1) if reason_match else "Unable to extract reasoning"
        )

        sufficient_match = re.search(r'"Sufficient":\s*"(.*?)"', string)
        sufficient = sufficient_match.group(1) if sufficient_match else "No"

        print("Answer:", answer)
        print("Reason:", reason)
        print("Sufficient:", sufficient)

        return answer, reason, sufficient


def break_question(question, args):
    """Break down question into sub-objectives"""
    from .prompt_list import subobjective_prompt

    prompt = subobjective_prompt + question
    response, token_num = run_llm(
        prompt,
        args.temperature_reasoning,
        args.max_length,
        args.openai_api_keys,
        args.LLM_type,
        False,
        False,
    )

    first_brace_p = response.find("[")
    last_brace_p = response.rfind("]")

    if first_brace_p != -1 and last_brace_p != -1:
        response = response[first_brace_p : last_brace_p + 1]

    return response, token_num


def save_2_jsonl(
    question,
    question_string,
    answer,
    cluster_chain_of_entities,
    call_num,
    all_t,
    start_time,
    file_name,
):
    """Save results to JSONL file"""
    tt = time.time() - start_time
    result_dict = {
        question_string: question,
        "results": answer,
        "reasoning_chains": cluster_chain_of_entities,
        "call_num": call_num,
        "total_token": all_t["total"],
        "input_token": all_t["input"],
        "output_token": all_t["output"],
        "time": tt,
    }

    with open(f"PoG_{file_name}.jsonl", "a") as outfile:
        json_str = json.dumps(result_dict)
        outfile.write(json_str + "\n")


def convert_dict_name(ent_rel_ent_dict, entid_name):
    """Convert entity IDs to names in the entity-relation-entity dictionary"""
    name_dict = {}
    for topic_e, h_t_dict in ent_rel_ent_dict.items():
        if entid_name.get(topic_e) not in name_dict:
            name_dict[entid_name.get(topic_e, topic_e)] = {}

        for h_t, r_e_dict in h_t_dict.items():
            if h_t not in name_dict[entid_name.get(topic_e, topic_e)]:
                name_dict[entid_name.get(topic_e, topic_e)][h_t] = {}

            for rela, e_list in r_e_dict.items():
                if rela not in name_dict[entid_name.get(topic_e, topic_e)][h_t]:
                    name_dict[entid_name.get(topic_e, topic_e)][h_t][rela] = []
                for ent in e_list:
                    ent_name = entid_name.get(ent, ent)
                    if (
                        ent_name
                        not in name_dict[entid_name.get(topic_e, topic_e)][h_t][rela]
                    ):
                        name_dict[entid_name.get(topic_e, topic_e)][h_t][rela].append(
                            ent_name
                        )

    return name_dict
