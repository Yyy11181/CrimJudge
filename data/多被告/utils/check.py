#检查id是否重复
import json
def check_id(id_list):
    id_set = set(id_list)
    if len(id_set) == len(id_list):
        return True
    else:
        return False
with open('JuDGE-main\data\多被告\多被告judge_new611.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    id_list = [item['Fact'] for item in data]
    print(check_id(id_list))
    #打印重复的
    for i in range(len(id_list)):
        if id_list.count(id_list[i]) > 1:
            print(id_list[i])