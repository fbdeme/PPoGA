import json

def read_output(file_path, question_string):
    answered_dict = {}
    file_path = '../../../results/'+file_path + '.jsonl'  # PPoGA 결과 디렉토리
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            data = json.loads(line)
            answered_dict[data[question_string]] = data

    answered_list = list(answered_dict.values())
    return answered_list

def prepare_dataset_for_eval(dataset_name, output_file):
    if dataset_name == 'cwq':
        with open('../../../data/cwq.json',encoding='utf-8') as f:  # PPoGA 데이터 디렉토리
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'webqsp':
        with open('../../../data/WebQSP.json',encoding='utf-8') as f:  # PPoGA 데이터 디렉토리
            datas = json.load(f)
        question_string = 'RawQuestion'
    elif dataset_name == 'grailqa':
        with open('../../../data/grailqa.json',encoding='utf-8') as f:  # PPoGA 데이터 디렉토리
            datas = json.load(f)
        question_string = 'question'
    else:
        print("dataset not found, you should pick from {cwq, webqsp, grailqa}.")
        exit(-1)

    output_datas = read_output(output_file, question_string)
    return datas, question_string, output_datas


def align(dataset_name, question_string, data, ground_truth_datas, aname_dict, alias_dict, add_ans_alias_dict):
    answer_list= []
    origin_data = [j for j in ground_truth_datas if j[question_string] == data[question_string]][0]
    if dataset_name == 'cwq':
        add_data = aname_dict[data[question_string]]
        add_ans_alias_data = add_ans_alias_dict[data[question_string]]
        add_data += add_ans_alias_data
        if 'answers' in origin_data:
            answers = origin_data["answers"]
        else:
            answers = origin_data["answer"]
        if answers not in add_data:
            add_data.append(answers)
        
        answer_list = add_data
        alias_list = []
        for x in answer_list:
            if x in alias_dict.keys():
                alias_list += alias_dict[x]
        
        answer_list = list(set(answer_list)|set(alias_list))

    elif dataset_name == 'webqsp':
        answers = origin_data["Parses"]
        for answer in answers:
            for name in answer['Answers']:
                if name['EntityName'] == None:
                    answer_list.append(name['AnswerArgument'])
                else:
                    answer_list.append(name['EntityName'])

        alias_list = []
        for x in answer_list:
            if x in alias_dict.keys():
                alias_list += alias_dict[x]
        
        answer_list = list(set(answer_list)|set(alias_list))

    elif dataset_name == 'grailqa':
        answers = origin_data["answer"]
        for answer in answers:
            if "entity_name" in answer:
                answer_list.append(answer['entity_name'])
            else:
                answer_list.append(answer['answer_argument'])

    return list(set(answer_list)), origin_data
    

def exact_match(response, answers):
    clean_result = response.strip().replace(" ","").lower()
    for answer in answers:
        clean_answer = answer.strip().replace(" ","").lower()
        if clean_result == clean_answer or clean_result in clean_answer or clean_answer in clean_result:
            return True
    return False
