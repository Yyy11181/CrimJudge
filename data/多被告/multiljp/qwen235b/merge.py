import os
import json


def merge_jsonl_files(directory_path, output_merged_jsonl_name="merged_output.jsonl"):
    # 合并指定目录下所有JSON文件到一个新的JSONL文件。
    merged_file_path = os.path.join(directory_path, output_merged_jsonl_name)
    json_files_found = []

    # 收集所有 .json 文件
    for filename in os.listdir(directory_path):
        if filename.endswith(".json") and filename != output_merged_jsonl_name:
            json_files_found.append(os.path.join(directory_path, filename))

    if not json_files_found:
        print(f"🤷 在 '{directory_path}' 目录中没有找到任何 .json 文件可供合并。")
        return

    merged_data = []
    # 合并文件
    for json_file in json_files_found:
        try:
            with open(json_file, 'r', encoding='utf-8') as infile:
                data = json.load(infile)
                merged_data.extend(data)
            print(f"➕ 已合并文件: '{json_file}'")
        except Exception as e:
            print(f"❌ 合并文件 '{json_file}' 时出错: {e}")

    # 将合并后的数据写入输出文件
    try:
        with open(merged_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(merged_data, outfile, ensure_ascii=False, indent=4)
        print(f"✅ 所有JSON文件已合并到 '{merged_file_path}'")
    except Exception as e:
        print(f"❌ 写入合并文件 '{merged_file_path}' 时出错: {e}")


merge_jsonl_files('data\多被告\multiljp\qwen235b', 'outputall.jsonl')