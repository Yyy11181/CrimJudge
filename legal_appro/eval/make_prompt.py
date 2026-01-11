from use_template import (
    use_reasoning_judge_template,
)
import logging
import json
import os
import argparse
data_len = 500
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
original_data_path = "data"

def make_prompt(original_data_path, generated_file_path, output_file_path, content_type):
    # original_file = open(original_data_path,'r', encoding='utf-8').readlines()
    #改为json.loads读取
    # with open(original_data_path, 'r', encoding='utf-8') as f:
    #     original_file = json.load(f)

    generated_file = []
    with open(generated_file_path, 'r', encoding='utf-8') as f:
        generated_file = json.load(f)  
    data_len = len(generated_file)
    if os.path.exists(output_file_path):
        os.remove(output_file_path)
        
    with open(output_file_path,'a', encoding='utf-8') as out_file:
        for id in range(0,data_len):
            # original_dict = json.loads(original_file[id])
            generated_dict = generated_file[id]
            court_review = generated_dict['exp_ans']
            if content_type == 'criminal_new':
                prompt = use_reasoning_judge_template(generated_dict['input'],generated_dict['Law_Articles_content'], court_review, generated_dict['gen_ans'])
            out_file.write(json.dumps({"id": generated_dict["id"],"type": content_type, "prompt": prompt}, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成提示信息")
    parser.add_argument("--model_name", type=str, help="评测对象模型", default='all')# farui_SLJA,gemini_SLJA,qwen235_SLJA,v3_SLJA
    parser.add_argument('--task_name', type=str, default='criminal_new', help='要处理的任务名称 (defense, fact, reasoning, judgement, criminal)，如果不指定则处理所有任务')
    args = parser.parse_args()
    model_name = args.model_name
    task_name = args.task_name

    tasks = ['defense', 'fact', 'reasoning', 'judgement', 'criminal_new']
    if task_name in tasks:
        tasks = [task_name] 

    # generated_dir = f"generate\\generated_data\\{model_name}"
    # output_dir = f"eval\\prompt\\{model_name}"
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    model_names = [name.strip() for name in args.model_name.split(',') if name.strip()]
    for model_name in model_names:
        generated_dir = f"legal_appro\\generate\\generated_data\\{model_name}"  # 生成的文件目录
        output_dir = f"eval\\prompt\\{model_name}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for task in tasks:
            logger.info(f"开始处理任务: {task} in {model_name}")

            generated_file_path = os.path.join(generated_dir, f"{task}.json")
            output_file_path = os.path.join(output_dir, f"{task}_eval_prompt.json")
            make_prompt(original_data_path, generated_file_path, output_file_path, task)


