Official repository of the paper: _Defense Against Backdoor Attacks on Neural Code Models via Source Code Purification_

### Project Summary
Neural code models have found widespread success in tasks pertaining to code intelligence, yet they are vulnerable to backdoor attacks, where an adversary can manipulate the victim model’s behavior by inserting triggers into the source code. Recent studies indicate that advanced backdoor attacks can achieve nearly 100% attack success rates on many software engineering tasks. However, effective defense techniques against such attacks remain insufficiently explored. In this study, we propose CodePurify, a novel defense method against backdoor attacks on code models through source code purification. Source code purification involves the process of precisely detecting and eliminating the possible triggers in the source code while preserving its semantic information. Within this process, CodePurify first develops a confidence-driven entropy-based measurement to determine whether a code snippet is poisoned and, if so, locates the triggers. Subsequently, it purifies the code by substituting the triggers with benign tokens using a masked language model. We extensively evaluate CodePurify against four advanced backdoor attacks across three representative tasks and two popular code models. The results show that CodePurify significantly outperforms three commonly used defense baselines, improving average defense performance by at least 41%, 68%, and 12% across the three tasks, respectively. These findings highlight the potential of CodePurify to serve as a robust defense against backdoor attacks on neural code models.

### Quick Tour
To reproduce the experimental results, you need to perform backdoor attacks on code models and then use our defense method, CodePurify, to protect the victim code models.
Specifically, you should first sample some data examples from datasets for poisoning. Then, select the baseline triggers and insert them into these data examples. The victim model will have a backdoor installed after being trained on the poisoned training set. Finally, apply our defense method, CodePurify, to protect the victim model from these attacks.  
Now, we provide an example of attacking and defending the CodeBERT model on the program repair task.

- Sample data examples from datasets for poisoning
```
cd program_repair/data
python preprocess.py
```
- Trigger generation and insertion
```
python poison_data.py
```
- Train the victim model
```
cd program_repair/src/codebert
python run_attack.py \
--do_train \
--do_eval \
--model_type=roberta \
--model_name_or_path=codebert_base \
--config_name=codebert_base \
--tokenizer_name=codebert_base \
--load_model_path=./saved_models/checkpoint-last/pytorch_model.bin \
--train_filename=./../../data/train_icpr22_fixed_0.05.jsonl \
--dev_filename=./../../data/valid.jsonl \
--attack_type 'icpr22_fixed_0.05' \
--output_dir=./saved_models/ \
--max_source_length 168 \
--max_target_length 168 \
--beam_size 1 \
--train_batch_size 32 \
--eval_batch_size 32 \
--learning_rate 2e-5 \
--train_steps 40000 \
--eval_steps 2000
```
- Run our defense CodePurify
```
cd program_repair/src/codebert
purification_defense.py
```
