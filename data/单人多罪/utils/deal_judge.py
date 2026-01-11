import json,random

def count_accusation(data):
        accusation_count = {}
        for item in data:
            accusation = item['Crime Type']
            for acc in accusation:
                if acc in accusation_count:
                    accusation_count[acc] += 1
                else:
                    accusation_count[acc] = 1
        return accusation_count

with open(r"JuDGE-main\data\单人多罪\judge_单人多罪612.jsonl", 'r', encoding='utf-8') as f:
    data1 = [json.loads(line) for line in f]

# with open(r"multiple_crimes_cases1.json", 'r', encoding='utf-8') as f:
#     data2 = json.load(f)

# data = data1 + data2
# random.seed(42)  # 固定随机种子保证可复现
# sampled_lines = random.sample(data, 500)

accusation_count = count_accusation(data1)

#保存到文件
with open(r'JuDGE-main\data\单人多罪\utils\accusation_count_judge.json', 'w', encoding='utf-8') as f:
    json.dump(accusation_count, f, ensure_ascii=False, indent=4)


# with open(r"data\单人多罪\judge_单人多罪612.json", 'w', encoding='utf-8') as f:
#         json.dump(sampled_lines, f, ensure_ascii=False, indent=4)
