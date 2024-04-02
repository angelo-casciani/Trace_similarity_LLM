from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, AutoConfig
from torch import cuda, bfloat16
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
import os
import datetime


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
        max_new_tokens=512,
        repetition_penalty=1.1
    )

    return generate_text


def log_to_file(message, curr_datetime):
    folder = 'tests'
    sub_folder = 'outputs'
    filename = f"output_{curr_datetime}.txt"
    filepath = os.path.join(folder, sub_folder, filename)
    with open(filepath, 'a') as file:
        file.write(message)


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


if __name__ == "__main__":
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    hf_token = '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'  # HuggingFace Token
    model_id_mistral = 'mistralai/Mistral-7B-Instruct-v0.2'
    model_id_llama7 = 'meta-llama/Llama-2-7b-chat-hf'
    # model_identifier = 'meta-llama/Llama-2-13b-chat-hf'

    pipeline_mistral = initialize_pipeline(model_id_mistral, hf_token)
    pipeline_llama7 = initialize_pipeline(model_id_llama7, hf_token)
    hf_pipeline_mistral = HuggingFacePipeline(pipeline=pipeline_mistral)
    hf_pipeline_llama7 = HuggingFacePipeline(pipeline=pipeline_llama7)

    template = """<s>[INST] {question} [/INST]"""
    template_task_description = """<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{question} [/INST]"""
    prompt = PromptTemplate.from_template(template)
    chain_mistral = prompt | hf_pipeline_mistral
    chain_llama7 = prompt | hf_pipeline_llama7

    live_prompting(chain_mistral, chain_llama7)
