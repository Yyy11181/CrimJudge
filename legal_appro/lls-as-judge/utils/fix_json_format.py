import json
import re

# 读取JSON文件内容
with open('matching_data.json', 'r', encoding='utf-8') as f:
    content = f.read()

# 使用正则表达式修复第1883行附近的exp_ans字段
# 查找exp_ans字段的开始和结束位置
pattern = r'"exp_ans":\s*"([^"]+)"'

# 尝试找到exp_ans字段
match = re.search(pattern, content)
if match:
    print("找到exp_ans字段，内容正常")
else:
    print("未找到正常的exp_ans字段，开始修复...")
    # 尝试找到异常的exp_ans字段
    pattern = r'"exp_ans":\s*([^,}]+)'
    match = re.search(pattern, content)
    if match:
        print("找到异常的exp_ans字段")
        start = match.start()
        end = match.end()
        # 提取字段内容
        field_content = content[start:end]
        print(f"异常字段内容: {field_content[:100]}...")
        
        # 修复字段，正确包裹双引号并转义内部引号
        # 找到真正的exp_ans值结束位置（下一个键开始）
        next_key_pattern = r'"[a-zA-Z_]+"\s*:'
        next_key_match = re.search(next_key_pattern, content[end:])
        if next_key_match:
            exp_ans_value = content[match.end():end + next_key_match.start()]
            # 转义内部引号
            escaped_value = exp_ans_value.replace('"', '\\"')
            # 移除可能的换行符
            escaped_value = escaped_value.replace('\n', '')
            # 构建修复后的字段
            fixed_field = f'"exp_ans": "{escaped_value}"' 
            # 替换异常字段
            fixed_content = content[:start] + fixed_field + content[end + next_key_match.start():]
            
            # 尝试解析修复后的JSON，验证是否有效
            try:
                json.loads(fixed_content)
                print("JSON修复成功，格式有效")
                # 保存修复后的文件
                with open('matching_data_fixed.json', 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                print("修复后的文件已保存为matching_data_fixed.json")
            except json.JSONDecodeError as e:
                print(f"JSON解析失败: {e}")
        else:
            print("无法找到exp_ans值的结束位置")
    else:
        print("未找到exp_ans字段")

print("修复完成")
