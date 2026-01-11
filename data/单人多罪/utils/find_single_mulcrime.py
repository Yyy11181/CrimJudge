import re
import json
from collections import defaultdict
crime_patterns = {
    "走私、贩卖、运输、制造毒品罪": r"走私、贩卖、运输、制造毒品罪|贩卖毒品罪|运输毒品罪|制造毒品罪",
    "盗窃罪": r"盗窃罪",
    "抢劫罪": r"抢劫罪",
    "容留他人吸毒罪": r"容留他人吸毒罪",
    "强奸罪": r"强奸罪",
    "故意伤害罪": r"故意伤害罪",
    "诈骗罪": r"诈骗罪",
    "寻衅滋事罪": r"寻衅滋事罪",
    "危险驾驶罪": r"危险驾驶罪",
    "非法持有毒品罪": r"非法持有毒品罪",
    "非法持有、私藏枪支、弹药罪": r"非法持有、私藏枪支、弹药罪",
    "敲诈勒索罪": r"敲诈勒索罪",
    "非法拘禁罪": r"非法拘禁罪",
    "妨害公务罪": r"妨害公务罪",
    "开设赌场罪": r"开设赌场罪",
    "职务侵占罪": r"职务侵占罪",
    "合同诈骗罪": r"合同诈骗罪",
    "抢夺罪": r"抢夺罪",
    "聚众斗殴罪": r"聚众斗殴罪",
    "信用卡诈骗罪": r"信用卡诈骗罪",
    "交通肇事罪": r"交通肇事罪",
    "故意毁坏财物罪": r"故意毁坏财物罪",
    "赌博罪": r"赌博罪",
    "非法侵入住宅罪": r"非法侵入住宅罪",
    "贪污罪": r"贪污罪",
    "受贿罪": r"受贿罪",
    "强制猥亵、侮辱罪": r"强制猥亵、侮辱罪",
    "放火罪": r"放火罪",
    "非法制造、买卖、运输、邮寄、储存枪支、弹药、爆炸物罪": r"非法制造、买卖、运输、邮寄、储存枪支、弹药、爆炸物罪",
    "以危险方法危害公共安全罪": r"以危险方法危害公共安全罪",
    "猥亵儿童罪": r"猥亵儿童罪",
    "故意杀人罪": r"故意杀人罪",
    "挪用公款罪": r"挪用公款罪",
    "非法吸收公众存款罪": r"非法吸收公众存款罪",
    "组织卖淫罪": r"组织卖淫罪",
    "引诱、容留、介绍卖淫罪": r"引诱、容留、介绍卖淫罪",
    "伪造、变造、买卖国家机关公文、证件、印章罪": r"伪造、变造、买卖国家机关公文、证件、印章罪",
    "挪用资金罪": r"挪用资金罪",
    "窝藏、包庇罪": r"窝藏、包庇罪",
    "制作、复制、出版、贩卖、传播淫秽物品牟利罪": r"制作、复制、出版、贩卖、传播淫秽物品牟利罪",
    "虚假出资、抽逃出资罪": r"虚假出资、抽逃出资罪",
    "贷款诈骗罪": r"贷款诈骗罪",
    "招摇撞骗罪": r"招摇撞骗罪",
    "强制猥亵、侮辱妇女罪": r"强制猥亵、侮辱妇女罪",
    "破坏电力设备罪": r"破坏电力设备罪",
    "行贿罪": r"行贿罪",
    "滥伐林木罪": r"滥伐林木罪",
    "组织、领导传销活动罪": r"组织、领导传销活动罪",
    "过失以危险方法危害公共安全罪": r"过失以危险方法危害公共安全罪",
    "侵犯著作权罪": r"侵犯著作权罪",
    "持有、使用假币罪": r"持有、使用假币罪",
    "出售、购买、运输假币罪": r"出售、购买、运输假币罪",
    "骗取贷款、票据承兑、金融票证罪": r"骗取贷款、票据承兑、金融票证罪",
    "妨害信用卡管理罪": r"妨害信用卡管理罪",
    "强迫交易罪": r"强迫交易罪",
    "妨害作证罪": r"妨害作证罪"
    # ... 其余罪名省略，使用相同模式添加
}
def find_single_defendant_multiple_crimes(case_text):
    """
    查找单一被告人的多罪案件，排除多个被告人的情况
    """
    # 首先检查是否存在多个被告人的模式
    multiple_defendants_patterns = [
        r"[一二三四五六七八九十]、\s*被告人",  # 匹配序号 + 被告人
        r"\d+、\s*被告人",                      # 匹配数字 + 被告人
        r"被告人.*?[，、；。].*?被告人"         # 匹配多个被告人
    ]
    
    # 如果符合任何一个多被告人模式，直接返回False
    for pattern in multiple_defendants_patterns:
        if re.search(pattern, case_text):
            return False, None, None
            
    # 查找单一被告人的多罪判决模式
    judgment_pattern = r"被告人([^，。；\s]{2,4})犯([^，。；]+)罪[，。；].*?判处.*?(?:；|。).*?犯([^，。；]+)罪"
    match = re.search(judgment_pattern, case_text)
    
    if match:
        defendant = match.group(1)
        crime1 = match.group(2) + "罪"
        crime2 = match.group(3) + "罪"
        
        # 验证罪名
        # if crime1 in crime_patterns and crime2 in crime_patterns:
        return True, defendant, sorted([crime1, crime2])
    
    return False, None, None

def process_cases():
    """处理案件并保存符合条件的数据"""
    multiple_crimes_cases = []
    crime_combinations = defaultdict(int)

    # 读取原始数据
    try:
        with open(r'JuDGE-main\data\单人单罪\judge_new1.jsonl', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]
            for case in data:
                try:
                    # case = json.loads(line.strip())
                    is_multiple, defendant, crimes = find_single_defendant_multiple_crimes(case['Judgment'])
                    
                    if is_multiple:
                        # 添加被告人和罪名信息到案件数据
                        case['defendant'] = defendant
                        case['crimes'] = crimes
                        multiple_crimes_cases.append(case)
                        
                        # 统计罪名组合
                        crime_key = ' + '.join(crimes)
                        crime_combinations[crime_key] += 1
                except json.JSONDecodeError as e:
                    print(f"JSON 解析出错: {str(e)}")
                except Exception as e:
                    print(f"处理案件时出错: {str(e)}")
    except FileNotFoundError:
        print("未找到原始数据文件")
    
    # 保存符合条件的案件
    try:
        with open(r'multiple_crimes_cases.json', 'w', encoding='utf-8') as f:
            json.dump(multiple_crimes_cases, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"保存案件数据时出错: {str(e)}")
    
    # 保存罪名组合统计
    try:
        with open(r'crime_combinations.json', 'w', encoding='utf-8') as f:
            json.dump(dict(crime_combinations), f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"保存罪名组合统计时出错: {str(e)}")
    
    # print(f"共找到 {len(multiple_crimes_cases)} 个单人多罪案件")
    # print(f"发现 {len(crime_combinations)} 种不同的罪名组合")
    # print("\n最常见的罪名组合：")
    # for combo, count in sorted(crime_combinations.items(), key=lambda x: x[1], reverse=True)[:5]:
    #     print(f"{combo}: {count}件")

if __name__ == "__main__":
    process_cases()