import os
import json

base_dir = os.path.dirname(__file__)
for folder in os.listdir(base_dir):
    if '_cmdl' in folder:
        json_path = os.path.join(base_dir, folder, 'criminal_new.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except Exception as e:
                    print(f"读取 {json_path} 失败: {e}")
                    continue
            for idx, item in enumerate(data, 1):
                item['id'] = idx
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"已为 {json_path} 每条数据添加id字段。")