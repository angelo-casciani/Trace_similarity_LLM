from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, AutoConfig
from torch import cuda, bfloat16
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
import os
import datetime

from oracle import AnswerVerificationOracle


def initialize_pipeline(model_id, hf_auth):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    model_config = AutoConfig.from_pretrained(
        model_id,
        token=hf_auth
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        token=hf_auth
    )
    # model.eval()
    # print(f"Model loaded on {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_auth
    )

    generate_text = pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=True,
        task='text-generation',
        do_sample=True,
        max_new_tokens=512,  # 4096 for traces generation
        repetition_penalty=1.1
    )

    return generate_text


def log_to_file(message, curr_datetime):
    folder = 'tests'
    sub_folder = 'outputs'
    filename = f"output_{curr_datetime}.txt"
    filepath = os.path.join(folder, sub_folder, filename)
    with open(filepath, 'a') as file1:
        file1.write(message)


def produce_answers(question, curr_datetime, chain1, chain2):
    answer1 = chain1.invoke({"question": question})
    print(f'Mistral\n: {answer1}\n')
    answer2 = chain2.invoke({"question": question})
    print(f'Llama: {answer2}')
    print('--------------------------------------------------')

    log_to_file(f'Query: {question}\n\nLlama2: {answer1}\n\nMistral:{answer2}\n\n\n', curr_datetime)


def live_prompting(model1, model2):
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    while True:
        query = input('Insert the query (type "quit" to exit): ')

        if query.lower() == 'quit':
            print("Exiting the chat.")
            break

        produce_answers(query, current_datetime, model1, model2)
        print()


def generate_altered_trace(question, chain1):
    complete_answer = chain1.invoke({"question": question})
    index = complete_answer.find('[/INST]')
    trace = complete_answer[index + len('[/INST]'):]
    with open('data/altered_traces_yes.txt', 'a') as file1:
        file1.write(trace)


def generate_wrong_trace(question, chain1):
    complete_answer = chain1.invoke({"question": question})
    index = complete_answer.find('[/INST]')
    trace = complete_answer[index + len('[/INST]'):]
    with open('data/altered_traces_no.txt', 'a') as file1:
        file1.write(trace)


def generate_traces(lang_chain, trace):
    question_yes = f"""Given this XES execution trace: {trace}
               Generate an altered version that has at least one alterations for its events among these ones:
                - use synonyms for the words for the value of the "concept:name" (e.g., 'recover info' becomes 'retrieve data');
                - add additional spaces between words for the value of the "concept:name" (e.g., 'recover info' becomes ' recover   info');
                - make some syntax errors for the value of the "concept:name" (e.g., 'recover info' becomes 'rcver info' or 'rwcover info').
                Do not add comments nor new tags  in the trace."""

    question_no = f"""Given this XES execution trace: {trace}
               Generate an altered version that has at least one alterations for its events among these ones:
                - use a different value for 'elementId' (e.g., 'Activity_0gh475d' becomes 'Activity_f2s733s');
                - use a different value for 'resourceId' (e.g., 'Lane_0f7txj4' becomes 'Lane_7sne8ys');
                - use a different value for 'org:resource' (e.g., 'Resource1-000001' becomes 'Resource1-000123');
                - use a different value for 'processId' (e.g., 'Process_1' becomes 'Process_1');
                - use a different value for 'time:timestamp' (e.g., '2023-07-05T16:24:55.304+02:00' becomes '2021-12-22T11:21:32.245+02:00');
                - remove an event of the trace;
                - remove a tag from an event.
                Do not add comments nor new tags in the trace."""

    for i in range(0, 100):
        print(f'Processing trace {i + 1} of 100')
        generate_altered_trace(question_yes, lang_chain)
    for i in range(0, 100):
        print(f'Processing trace {i + 1} of 100')
        generate_wrong_trace(question_no, lang_chain)


def build_trace_dictionary(file_path, expected_answer):
    trace_dict = {}
    current_trace = ""
    with open(file_path, 'r') as file1:
        for line in file1:
            if "<trace>" in line:
                current_trace = line.strip()
            elif "</trace>" in line:
                current_trace += line.strip()
                if current_trace:
                    trace_dict[current_trace] = expected_answer
            else:
                current_trace += line.strip()
    return trace_dict


def evaluate_chain_one_shot(eval_oracle, lang_chain, reference, traces_dict):
    count = 0
    for trace, exp_ans in traces_dict.items():
        question = f"""Trace 1: {reference}
                        Trace 2: {trace}
                        Does trace 1 and trace 2 correspond to the same trace? Provide only a binary response (i.e., 'yes' or 'no') with no further explanations."""
        eval_oracle.add_prompt_expected_answer_pair(question, exp_ans)
        complete_answer = lang_chain.invoke({"question": question})
        index = complete_answer.find('[/INST]')
        answer = complete_answer[index + len('[/INST]'):]
        eval_oracle.verify_answer(answer, question)
        count += 1
        print(f'Processing answer for trace {count} of {len(traces_dict)}...')

    print('Validation process completed. Check the output file.')
    eval_oracle.write_results_to_file()


def evaluate_chain_few_shots(eval_oracle, lang_chain, reference, traces_dict):
    trace_yes = ''
    trace_no = ''
    for key, value in traces_dict.items():
        if value == 'yes':
            trace_yes = key
        else:
            trace_no = key
        if trace_yes != '' and trace_no != '':
            break

    count = 0
    for trace, exp_ans in traces_dict.items():
        question = f"""Does trace 1 and trace 2 correspond to the same trace? Provide only a binary response (i.e., 'yes' or 'no') with no further explanations.
                        Question 1: 
                                Trace 1: {reference}
                                Trace 2: {trace_yes}
                        Answer 1: 'yes'
                        Question 2: 
                                Trace 1: {reference}
                                Trace 2: {trace_no}
                        Answer 2: 'no'
                        Question 3:
                                Trace 1: {reference}
                                Trace 2: {trace}
                        Answer 3: """

        eval_oracle.add_prompt_expected_answer_pair(question, exp_ans)
        complete_answer = lang_chain.invoke({"question": question})
        index = complete_answer.find('[/INST]')
        answer = complete_answer[index + len('[/INST]'):]
        eval_oracle.verify_answer(answer, question)
        count += 1
        print(f'Processing answer for trace {count} of {len(traces_dict)}...')

    print('Validation process completed. Check the output file.')
    eval_oracle.write_results_to_file()


def evaluate_chain_task_description(eval_oracle, lang_chain, reference, traces_dict):
    count = 0
    for trace, exp_ans in traces_dict.items():
        question = f"""Trace 1: {reference}
                        Trace 2: {trace}
                        Does trace 1 and trace 2 correspond to the same trace? Provide only a binary response (i.e., 'yes' or 'no') with no further explanations."""
        sys_message = "You are an expert system in examining XES execution traces of business processes. Two traces are the same if they are identical or if the attributes of the events compising the traces have the same values except for some slight modifications of the concept:name, otherwise you have to consider them different."
        eval_oracle.add_prompt_expected_answer_pair(question, exp_ans)
        complete_answer = lang_chain.invoke({"question": question, "system_message": sys_message})
        index = complete_answer.find('[/INST]')
        answer = complete_answer[index + len('[/INST]'):]
        eval_oracle.verify_answer(answer, question)
        count += 1
        print(f'Processing answer for trace {count} of {len(traces_dict)}...')

    print('Validation process completed. Check the output file.')
    eval_oracle.write_results_to_file()


if __name__ == "__main__":
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    hf_token = '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'  # HuggingFace Token
    model_id = ''

    while model_id == '':
        model_choice = input(
            "Select the desired LLM to test:\n1. Mistral 7B;\n2. Llama 2 7B;\n3. Llama 2 13B.\nYour choice: ")
        if model_choice == '1':
            model_id = 'mistralai/Mistral-7B-Instruct-v0.2'
        elif model_choice == '2':
            model_id = 'meta-llama/Llama-2-7b-chat-hf'
        elif model_choice == '3':
            model_id = 'meta-llama/Llama-2-13b-chat-hf'
        else:
            print("Not a valid choice!")

    pipeline = initialize_pipeline(model_id, hf_token)
    hf_pipeline = HuggingFacePipeline(pipeline=pipeline)

    template = ''
    strategy = ''
    while template == '':
        strategy = input(
            "Select the desired prompting strategy:\n1. One-shot;\n2. Few-shots;\n3. Task description.\nYour choice: ")
        if strategy == '1':
            template = """<s>[INST] {question} [/INST]"""
        elif strategy == '2':
            template = """<s>[INST] {question} [/INST]"""
        elif strategy == '3':
            template = """<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{question} [/INST]"""
        else:
            print("Not a valid choice!")

    prompt = PromptTemplate.from_template(template)
    chain = prompt | hf_pipeline

    with open('data/ref_trace.txt', 'r') as file:
        ref_trace = file.read()
    file_path_yes = "data/altered_traces_yes.txt"
    file_path_no = "data/altered_traces_no.txt"
    alt_traces_dict = build_trace_dictionary(file_path_yes, 'yes')
    alt_traces_dict.update(build_trace_dictionary(file_path_no, 'no'))

    oracle = AnswerVerificationOracle()
    if strategy == '1':
        evaluate_chain_one_shot(oracle, chain, ref_trace, alt_traces_dict)
    elif strategy == '2':
        evaluate_chain_few_shots(oracle, chain, ref_trace, alt_traces_dict)
    elif strategy == '3':
        evaluate_chain_task_description(oracle, chain, ref_trace, alt_traces_dict)

