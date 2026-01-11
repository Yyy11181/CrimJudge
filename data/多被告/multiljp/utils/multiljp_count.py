import json,random

with open("data\多被告\multiljp\multiljp.jsonl", "r", encoding="utf-8") as f:
    all_data = [json.loads(line) for line in f]

accusations_set = set()
laws_set = set()

random.seed(42)  # 固定随机种子保证可复现
sampled_lines = random.sample(all_data, 500)

# 遍历每个案件
accusation_count = {}
for data in sampled_lines:
    # 遍历每个被告，提取罪名和法条并加入集合
    for defendant, info in data['criminals_info'].items():
        #统计罪名的数量
        for accusation in info['accusations']:
            if accusation in accusation_count:
                accusation_count[accusation] += 1
            else:
                accusation_count[accusation] = 1
        accusations_set.update(info['accusations'])
        laws_set.update(info['laws'])

# 创建一个字典保存结果
output = {
    "accusations": list(accusations_set),
    "laws": list(laws_set)
}

with open('data\多被告\multiljp\crime_law.json', 'w', encoding='utf-8') as json_file:
    json.dump(output, json_file, ensure_ascii=False, indent=4)

with open(r'data\多被告\multiljp\accusation_count.json', 'w', encoding='utf-8') as f:
    json.dump(accusation_count, f, ensure_ascii=False, indent=4)

with open(r"data\多被告\multiljp\multiljp_new.jsonl", 'w', encoding='utf-8') as f:
    for line in sampled_lines:
        f.write(json.dumps(line, ensure_ascii=False) + '\n')
        # json.dump(sampled_lines, f, ensure_ascii=False, indent=4)