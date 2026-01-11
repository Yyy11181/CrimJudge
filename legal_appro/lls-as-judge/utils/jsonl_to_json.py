import json
import os

def convert_jsonl_to_json(jsonl_file_path, json_file_path):
    """
    将JSONL文件转换为JSON文件
    :param jsonl_file_path: JSONL文件路径
    :param json_file_path: 输出JSON文件路径
    """
    try:
        # 读取JSONL文件
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 解析每行JSON
        data = []
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if line:
                try:
                    json_obj = json.loads(line)
                    data.append(json_obj)
                except json.JSONDecodeError as e:
                    print(f"第{line_num}行解析失败: {e}")
        
        # 写入JSON文件
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"转换成功！")
        print(f"JSONL文件行数: {len(lines)}")
        print(f"成功解析行数: {len(data)}")
        print(f"JSON文件已保存到: {json_file_path}")
        print(f"JSON文件大小: {os.path.getsize(json_file_path)} 字节")
        
    except Exception as e:
        print(f"转换失败: {e}")

if __name__ == "__main__":
    # 输入和输出文件路径
    input_jsonl = r'lls-as-judge\criminal_new.jsonl'
    output_json = r'lls-as-judge\criminal_new.json'
    
    # 执行转换
    convert_jsonl_to_json(input_jsonl, output_json)