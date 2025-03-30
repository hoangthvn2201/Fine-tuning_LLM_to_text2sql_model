# Fine-tuning_LLM_to_text2sql_model
 
## About The Project 
Fine-tuning a Large Language Model to Text2SQL Model with the function of generating SQL query based on user database.

## Model Details
* Base Model: [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) 
* Learning rate: 2e-4
* lora_rank: 16
* lora_alpha: 32
* lora_trainable: 'up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj'
* lora_dropout: 0.05
* fp16: False
* max_seq_length: True
* num_train_epochs: 25
* early_stopping_patience: 5

## Dataset Details
* Link: [huyhoangt2201/contextawareJidouka_fixed](https://huggingface.co/datasets/huyhoangt2201/multitableJidouka_new_fixed_error)
* Number of records: 977
* Attributes:
  - previous question
  - previous answer: previous SQL answer or natural answer
  - schema linking: of preivious answer if answer is sql query (Format: [Tables, Columns, Foreign keys, Possible cell values])
  - question: related to previous question
  - answer: SQL query or natural answer
* Strategy for Creating Dataset: synthetic data using ChatGPT ([prompt gen dataset]())

## Use with transformers
   ```python
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from typing import Union, List, Dict, Any, Optional

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'huyhoangt2201/llama3.2_1b_finetuned_SQL_multitableJidouka'

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    system_prompt = """You are an SQL query assistant. Based on the table information below and conversation history, generate an SQL query to retrieve the relevant information for the user. Handle context-dependent questions by referring to previous conversation turns. If the user's question is unrelated to the table, respond naturally in user's language.
    The table contains the following columns: ...
    """ 
    question = "Cải tiến nào có sản phẩm đầu ra là file csv và tác giả là Nguyễn Văn A?"
    messages = [
        {'role':'system','content':system_prompt},
        {'role':'user','content':question}
    ]

    eot = "<|eot_id|>"
    eot_id = tokenizer.convert_tokens_to_ids(eot)
    tokenizer.pad_token = eot
    tokenizer.pad_token_id = eot_id

    messages_chat = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    inputs = tokenizer(
            messages_chat,
            return_tensors='pt',
            padding=True,
            truncation=True
    ).to(device)

    outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=True,
            top_p = 0.1
    ).to(device)

    response = tokenizer.decode(outputs[0])
    response = response.split('<|start_header_id|>assistant<|end_header_id|>')[1].strip()[:-10]

   ```
## Project Detail
Read more about project in [project_summary](https://github.com/hoangthvn2201/Fine-tuning_LLM_to_text2sql_model/blob/main/project_summary_29_12_2024.pdf)
