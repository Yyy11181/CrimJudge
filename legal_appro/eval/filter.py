import json

# 先过滤掉包含error的行，输出为一个临时文件
input_path = r'eval\llm_eval_r1_711_Judge\farui_judge\criminal_new.json'
tmp_path = r'eval\llm_eval_r1_711_Judge\farui_judge\criminal_new_noerror.json'

with open(input_path, 'r', encoding='utf-8') as fin, open(tmp_path, 'w', encoding='utf-8') as fout:
    for line in fin:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if 'response' in obj and 'error' in obj['response'].lower():
                continue  # 跳过包含error的行
            fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"跳过无法解析的行: {e}")
print(f"[1/2] 已过滤error并保存到 {tmp_path}")

# 再收集无error数据的id
filter_ids = set()
with open(tmp_path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if 'id' in obj:
                filter_ids.add(obj['id'])
        except Exception as e:
            print(f"[收集ID] 跳过无法解析的行: {e}")
print(f"[2/2] 共收集到 {len(filter_ids)} 个id")

# 用这些id过滤生成数据
gen_input_path = r'eval\prompt\farui_judge\criminal_new_eval_prompt.json'
gen_output_path = r'eval\prompt\farui_judge\criminal_new_eval_prompt_error.json'

with open(gen_input_path, 'r', encoding='utf-8') as fin, open(gen_output_path, 'w', encoding='utf-8') as fout:
    for line in fin:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if obj.get('id') in filter_ids:
                continue  # 跳过这些id
            fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"[生成数据-按ID过滤] 跳过无法解析的行: {e}")
print(f"[生成数据-按ID过滤] 已过滤并保存到 {gen_output_path}")
