import json
from data_segment_xingshi import DataSegmentXingshi
parser = DataSegmentXingshi(punctuation_replace=True)

with open('/home/swh/ybq/casegen/process/input/multi/vx/qwen2.5-7B-Instruct.json') as file:
    data = json.load(file)

for item in data:
    # wenshu = item["Full Document"]
    wenshu = item['gen_ans']
    result = parser.parse(wenshu)
    for i in result:
        print(i, result[i])
        print('-'*50)
    
    print('*'*150)