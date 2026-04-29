import csv
import json
from tqdm import tqdm
import random
import torch

min_log = 20
N_STU, N_EXERCISE, N_CONCEPT = (91905, 835, 41)


def divide_dict(dictionary: dict):
    list_d = list(dictionary.items())

    random.shuffle(list_d)
    total_items = len(list_d)
    n_train, n_val = int(total_items * 0.7), int(total_items * 0.1)
    train_dict = dict(list_d[:n_train])
    val_dict = dict(list_d[n_train:n_train+n_val])
    test_dict = dict(list_d[n_train+n_val:])

    return train_dict, val_dict, test_dict


def divide_data():
    problem_code = {}
    concept_code = {}
    stu_code = {}

    problem_to_concept = {}
    stu_log = {}

    with open('junyi_Exercise_table.csv', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)

        for row in tqdm(csv_reader):
            problem_name = row[0]
            concept_name = row[9]
            if problem_name not in problem_code.keys():
                p_id = len(problem_code.keys())
                problem_code[problem_name] = p_id
            else:
                p_id = problem_code[problem_name]

            if concept_name not in concept_code.keys():
                c_id = len(concept_code.keys())
                concept_code[concept_name] = c_id
            else:
                c_id = concept_code[concept_name]
            problem_to_concept[p_id] = c_id

        print(problem_code)
        print(concept_code)
        print(problem_to_concept)
        print(len(problem_code.keys()))
        print(len(concept_code.keys()))
        print()

        Q_matrix = torch.zeros((len(problem_code.keys()), len(concept_code.keys())))
        for p_id, c_id in problem_to_concept.items():
            Q_matrix[p_id, c_id] = 1
        torch.save(Q_matrix, 'Q-matrix.pt')

    with open('junyi_ProblemLog_original.csv', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)

        for row in tqdm(csv_reader):
            stu_name = row[0]
            if row[1] not in problem_code.keys():
                continue
            p_id = problem_code[row[1]]
            response = 0.0 if row[10] == "false" else 1.0

            if stu_name not in stu_code.keys():
                stu_id = len(stu_code.keys())
                stu_code[stu_name] = stu_id
                stu_log[stu_id] = {}
            else:
                stu_id = stu_code[stu_name]

            if response == 1:
                stu_log[stu_id][p_id] = 1
            else:
                stu_log[stu_id][p_id] = 0

        print("stu_num", len(stu_log.keys()))
        filtered_stu_log = {key: value for key, value in stu_log.items() if len(value) >= min_log}
        filtered_stu_log = {index: value for index, (_, value) in enumerate(filtered_stu_log.items())}
        print(filtered_stu_log)

        with open('log.json', 'w', encoding='utf8') as output_file:
            json.dump(filtered_stu_log, output_file, indent=4, ensure_ascii=False)



if __name__ == "__main__":
     divide_data()



