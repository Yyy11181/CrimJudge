import re,json

def find_cases_with_two_different_defendants(case_text):
    defendant_names = set()
    
    # 更精确的正则匹配模式
    patterns = [
        # 匹配"被告人 姓名"格式，使用负向前瞻避免匹配到动词
        r"被告人[\s:：]*([^\s，；。\n犯的之回]{2,3}(?:某|×)*)(?=[犯的之\s，。；]|$)",
        # 匹配序号格式：一、被告人 姓名
        r"(?:^|\n)[一二三四五六七八九十]、[\s]*被告人[\s:：]*([^\s，；。\n犯的之回]{2,3}(?:某|×)*)(?=[犯的之\s，。；]|$)",
        # 匹配数字序号格式：1、被告人 姓名
        r"(?:^|\n)\d+、[\s]*被告人[\s:：]*([^\s，；。\n犯的之回]{2,3}(?:某|×)*)(?=[犯的之\s，。；]|$)",
    ]
    
    # 使用多个正则模式进行匹配
    for pattern in patterns:
        matches = re.findall(pattern, case_text, re.MULTILINE)
        for match in matches:
            if isinstance(match, tuple):
                name = next((m for m in match if m), "").strip()
            else:
                name = match.strip()
            
            if name:
                # 清理姓名
                clean_name = re.sub(r"[（(].*?[）)]", "", name)
                clean_name = re.sub(r"[，、；。].*", "", clean_name)
                clean_name = clean_name.strip()
                
                # 验证姓名格式：2-3个字符，可以是中文、某、×
                if re.match(r"^[\u4e00-\u9fa5某×]{2,3}$", clean_name):
                    defendant_names.add(clean_name)
    
    # 使用更严格的模式来匹配被告人
    # 只匹配明确的"被告人+姓名+明确分隔符/动词"的格式
    strict_patterns = [
        r"被告人[\s]*([^\s，；。\n]{2,3}(?:某|×)*)犯",  # 被告人XXX犯
        r"被告人[\s]*([^\s，；。\n]{2,3}(?:某|×)*)的",  # 被告人XXX的（但要避免动作）
        r"被告人[\s]*([^\s，；。\n]{2,3}(?:某|×)*)，",  # 被告人XXX，
        r"被告人[\s]*([^\s，；。\n]{2,3}(?:某|×)*)[，。；]",  # 被告人XXX + 标点
        r"[一二三四五六七八九十]、[\s]*被告人[\s]*([^\s，；。\n]{2,3}(?:某|×)*)[犯，。；]",  # 序号格式
    ]
    
    for pattern in strict_patterns:
        matches = re.findall(pattern, case_text)
        for match in matches:
            clean_name = match.strip()
            if re.match(r"^[\u4e00-\u9fa5某×]{2,3}$", clean_name):
                defendant_names.add(clean_name)
    
    # 额外验证：排除可能的误匹配
    # 检查上下文，排除不合理的组合
    validated_names = set()
    for name in defendant_names:
        # 检查该姓名在文本中的上下文
        contexts = re.findall(rf"被告人[\s]*{re.escape(name)}(.{{0,3}})", case_text)
        
        valid = True
        for context in contexts:
            # 如果紧跟的是明显的动词（如"回到"、"逃跑"等），则可能是误匹配
            if re.match(r"^[回逃跑走去来到]", context):
                valid = False
                break
        
        if valid:
            validated_names.add(name)
    
    # 进一步清理：检查是否有相似的姓名
    cleaned_defendants = set()
    defendants_list = list(validated_names)
    
    for name in defendants_list:
        is_similar = False
        for existing_name in list(cleaned_defendants):
            # 如果一个姓名是另一个的前缀或后缀，认为可能是同一人
            if (name != existing_name and 
                (name.startswith(existing_name) or existing_name.startswith(name) or
                 name.endswith(existing_name) or existing_name.endswith(name))):
                is_similar = True
                # 保留较短的姓名（通常是正确的核心姓名）
                if len(name) < len(existing_name):
                    cleaned_defendants.remove(existing_name)
                    cleaned_defendants.add(name)
                elif len(name) == len(existing_name):
                    # 如果长度相同，保留不包含动作词的那个
                    if not re.search(r"[回走去来到跑逃]", name) and re.search(r"[回走去来到跑逃]", existing_name):
                        cleaned_defendants.remove(existing_name)
                        cleaned_defendants.add(name)
                break
        
        if not is_similar:
            cleaned_defendants.add(name)
    
    unique_defendants = list(cleaned_defendants)
    has_two_different = len(unique_defendants) >= 2
    
    return has_two_different, sorted(unique_defendants)

# 示例用法
with open(r"JuDGE-main\data\单人单罪\judge_new1.jsonl",encoding="utf-8") as f:
    data = [json.loads(item) for item in f]
find_cases = []
for item in data:
    case_text = item["Judgment"]
    has_two_defendants, defendant_names = find_cases_with_two_different_defendants(case_text)
    if has_two_defendants:
        print(f"案件内容：{case_text}")
        print(f"包含两个不同的被告人：{defendant_names}")
        print(f"案件id：{item['CaseId']}")
        item["defendant_names"] = defendant_names
        find_cases.append(item)
print(len(find_cases))
# # 写入JSON文件
# with open(r"JuDGE-main\data\多被告\612.json", "w", encoding="utf-8") as f:
#     # 转换为JSON Lines格式（JSONA1标准），每行一个JSON对象
#     json.dump(find_cases, f, ensure_ascii=False, indent=4)

# #更新之前的data\单人多罪\单人多罪.json
# for item in data:
#     if item in find_cases:
#         data.remove(item)
# with open(r"JuDGE-main\data\单人单罪\SLJA_cor_5001.json", "w", encoding="utf-8") as f:
#     # 转换为JSON Lines格式（JSONA1标准），每行一个JSON对象
#     json.dump(data, f, ensure_ascii=False, indent=4)