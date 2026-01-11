import re

def load_template(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

reasoning_template_path = r'legal_appro\eval\template\evaluate.txt'

reasoning_template = load_template(reasoning_template_path)



def use_reasoning_judge_template(fact, article, court_review, generated_reasoning):
    # while '<' in generated_reasoning:
    #     generated_reasoning = re.sub(r"<antThinking>.*?</antThinking>", "", generated_reasoning,flags=re.S)
    #     generated_reasoning = re.sub(r"<.*?>", "", generated_reasoning,flags=re.S)
    
    # 处理 article 参数为列表的情况
    if isinstance(article, list):
        # 如果列表中是字典，需要提取article_id和content
        article_texts = []
        for item in article:
            if isinstance(item, dict):
                # article_id = str(item.get('article_id', ''))
                content = str(item.get('content', ''))
                article_name= str(item.get('name', ''))
                # 组合article_id和content
                article_texts.append(f"{article_name}：{content}")
            else:
                article_texts.append(str(item))
        article = '\n'.join(article_texts)
    elif isinstance(article, dict):
        # 如果直接传入的是字典
        article_id = str(article.get('article_id', ''))
        content = str(article.get('content', ''))
        article = f"《中华人民共和国刑法》第{article_id}条：{content}"
    else:
        # 确保article是字符串
        article = str(article)
    
    prompt = reasoning_template.replace("{case_facts}", fact).replace("{articles_of_law}", article).replace("{generated_judgment_text}", generated_reasoning).replace("{court_review}", court_review)
    return prompt

