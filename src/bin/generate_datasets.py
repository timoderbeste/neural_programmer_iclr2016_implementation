import json
import argparse
import src.data_generators.dataset_generator as dg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate dataset of a specific type.')
    parser.add_argument('template_type', type=str, help='The type of the template to generate. Supported ones are: single_column.')
    parser.add_argument('size', type=int, help='The size of the dataset to be generated.')
    parser.add_argument('save_file_path', type=str, help='The name of the file to store the dataset.')
    args = parser.parse_args()

    if args.template_type == 'single_column':
        tables, data = dg.generate_single_column_table_dataset(args.size, 1, 20)
        question_dicts = []
        for table_idx, question, answer in data:
            table = tables[table_idx]
            question_dict = dict()
            question_dict['table'] = table
            question_dict['question'] = question
            question_dict['answer'] = answer[0]
            question_dict['answer_type'] = answer[1]
            question_dicts.append(question_dict)

        with open(args.save_file_path, 'w') as fp:
            json.dump(question_dicts, fp)

    else:
        print('Template type not found.')
        exit(2)
