import json

# 读取文件
file_path = r'data\多被告场景.json'
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    # data = [json.loads(line) for line in f]

# 初始化分类结果
simple_case = []
complex = []
others = []

# 遍历数据进行分类
for item in data:
    crime_types = item['Crime Type']
    # sentences = item['Sentence']
    # Fine = item['Fine']
    # 假设从判决内容中可以推测人数，这里简单根据罪名数量和刑期数量判断
    if len(crime_types) ==1 : #单人单罪
        simple_case.append(item)
    elif len (crime_types) > 1: #
        complex.append(item)
    else: #多人多罪
        others.append(item)


#剩余的数据保留下来，写入文档
remaining_data = [item for item in data if item not in simple_case + complex + others]
# 将剩余的数据写入新的 JSON 文件
output_path = r'data\多被告.jsonl'
with open(output_path, 'w', encoding='utf-8') as f:
    for item in remaining_data:
        f.write(json.dumps(item, ensure_ascii=False, indent=2) + '\n')
# 输出结果

print("simple_case:")
print(len(simple_case))


print("\ncomplex:")
print(len(complex))


print("\nothers:")
print(len(others))

#剩余的数据保留下来

# 如果你想将分类结果保存到新的 JSON 文件中
output_path = r'data\new_被告.json'
classified_data = {
    "simple_case": simple_case,
    "complex": complex,
    "others": others
}
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(classified_data, f, ensure_ascii=False, indent=2)

# # 将不同类型的数据分别写入不同的 JSONL 文件
base_path = r'data\多被告'

# # 单人单罪
# with open(f'{base_path}/single_person_single_crime.jsonl', 'w', encoding='utf-8') as f:
#     for item in single_person_single_crime:
#         f.write(json.dumps(item, ensure_ascii=False) + '\n')
# 单人单罪
with open(f'{base_path}/simple_case.json', 'w', encoding='utf-8') as f:
    json.dump(simple_case, f, ensure_ascii=False, indent=2)


# 单人多罪
with open(f'{base_path}/complex.json', 'w', encoding='utf-8') as f:
    json.dump(complex, f, ensure_ascii=False, indent=2)

    # for item in single_person_multiple_crime:
    #     f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 多人多罪
with open(f'{base_path}/other.json', 'w', encoding='utf-8') as f:
    json.dump(others, f, ensure_ascii=False, indent=2)

    # for item in multiple_persons_multiple_crime:
    #     f.write(json.dumps(item, ensure_ascii=False) + '\n')



# 统计每个文件中的罪名分布，按照降序的顺序排列
# def count_crime_types(file_path):
#     crime_type_counts = {}
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             item = json.loads(line)
#             crime_types = item['Crime Type']
#             for crime_type in crime_types:
#                 if crime_type in crime_type_counts:
#                     crime_type_counts[crime_type] += 1
#                 else:
#                     crime_type_counts[crime_type] = 1
#     return dict(sorted(crime_type_counts.items(), key=lambda x: x[1], reverse=True))

# # 统计单人单罪文件中的罪名分布
# single_person_single_crime_path = f'{base_path}/single_person_single_crime.jsonl'
# single_person_single_crime_counts = count_crime_types(single_person_single_crime_path)  # 统计单人单罪文件中的罪名分布
 
# # 统计单人多罪文件中的罪名分布
# single_person_multiple_crime_path = f'{base_path}/single_person_multiple_crime.jsonl'
# single_person_multiple_crime_counts = count_crime_types(single_person_multiple_crime_path)  # 统计单人多罪文件中的罪名分布


# # 统计多人多罪文件中的罪名分布
# multiple_persons_multiple_crime_path = f'{base_path}/multiple_persons_multiple_crime.jsonl'
# multiple_persons_multiple_crime_counts = count_crime_types(multiple_persons_multiple_crime_path)  # 统计多人多罪文件中的罪名分布

# # 都一起写入文档，按照降序的顺序排列
# with open(f'{base_path}/crime_type_counts.txt', 'w', encoding='utf-8') as f:
#     f.write("单人单罪文件中的罪名分布:\n")
#     f.write(str(single_person_single_crime_counts) + '\n\n')
#     f.write("单人多罪文件中的罪名分布:\n")
#     f.write(str(single_person_multiple_crime_counts) + '\n\n')
#     f.write("多人多罪文件中的罪名分布:\n")
#     f.write(str(multiple_persons_multiple_crime_counts) + '\n\n')

