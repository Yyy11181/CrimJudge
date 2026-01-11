import json
import argparse
from crime_extraction import get_crime
from judge_extraction import calc_time_sum
from law_extraction import get_penalcode_index_from_text

def save_valid_predictions():
    #读取生成结果文件
    input_file = r"data\多被告\judge\法睿\output.jsonl"  #生成结果文件
    output_file = r'data\多被告\judge\法睿\output1.jsonl'         # 有效预测结果输出文件，换个名字就行
    
    # 读取生成结果文件，找出有效预测结果的样本
    valid_predictions = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            gen_ans = data['gen_ans']
            # 获取生成的预测结果
            gen_crime, gen_time, _, gen_penalcode_index = get_crime(gen_ans), calc_time_sum(gen_ans), None, get_penalcode_index_from_text(gen_ans)
            
            # 检查是否是有效预测结果(存在风险啊啊)
            if len(gen_crime) > 0 or len(gen_penalcode_index) > 0 or gen_time != -1:
                valid_predictions.append(data)
    
    print(f"找到 {len(valid_predictions)} 个有效预测结果的样本")
    
    # 将有效预测结果写入新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in valid_predictions:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"已将有效预测结果写入 {output_file}")

def filter_original_samples():
    #读取原始数据文件
    orig_file = r'data\单人多罪\judge\judge_final.jsonl'    # 原始输入文件
    input_file = r'data\单人多罪\judge\qwen235\output.jsonl'  # 生成结果文件
    output_file = r'data\单人多罪\judge\qwen235\min.jsonl'

    
    # 读取生成结果文件，找出无预测结果的样本ID
    no_prediction_ids = set()
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            gen_ans = data['gen_ans']
            # 获取生成的预测结果
            gen_crime, gen_time, _, gen_penalcode_index = get_crime(gen_ans), calc_time_sum(gen_ans), None, get_penalcode_index_from_text(gen_ans)
            
            # 检查是否是无预测结果
            if len(gen_crime) == 0 and len(gen_penalcode_index) == 0 and gen_time == -1:
                no_prediction_ids.add(data['input'])  # 使用input作为ID
    
    print(f"找到 {len(no_prediction_ids)} 个无预测结果的样本")
    
    # 从原始文件中提取这些样本
    filtered_samples = []
    with open(orig_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if data['Fact'] in no_prediction_ids:
                filtered_samples.append(data)
    
    print(f"成功找到 {len(filtered_samples)} 个原始样本")
    
    # 将过滤后的样本写入新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in filtered_samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')
    # 保留有效的原始数据
    
    print(f"已将过滤后的样本写入 {output_file}")

if __name__ == "__main__":
    save_valid_predictions()
    # filter_original_samples()
