import json

file_path = r'data\单人单罪\O3\cail单人单罪_new\output_new.jsonl'

with open(file_path, 'r', encoding='utf-8') as f:
    # data = json.load(f) # 
    data = [json.loads(line) for line in f]   

filtered_data = [item for item in data if '抱歉' not in item.get('gen_ans')]


existing_samples = set(line['input'].strip() for line in filtered_data)

input_file = r'data\单人单罪\cail单人单罪_new0.jsonl' #这个是过滤的数据集


# 过滤没有跑的数据
with open(input_file, 'r', encoding='utf-8') as in_f, open(r'data\单人单罪\cail\qwen235\min.jsonl', 'w', encoding='utf-8') as out_f:
    for line in in_f:
        if json.loads(line)['Fact'].strip() not in existing_samples:
            out_f.write(line)

#把filtered_data写入文件
with open(r'data\单人单罪\cail\qwen235\actual.jsonl', 'w', encoding='utf-8') as f:
    for item in filtered_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')