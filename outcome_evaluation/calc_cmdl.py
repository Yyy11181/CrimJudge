import json
import argparse
import os
from crime_extraction import get_crime
from judge_extraction import calc_time_sum, calc_amt_sum
from law_extraction import get_penalcode_index_from_text
import re

class MetricsCalculator:
    def __init__(self, gen_file, exp_file):
        self.gen_file = gen_file
        self.exp_file = exp_file
        self.gen_data = self.load_data(gen_file)
        self.exp_data = self.load_data(exp_file)
        
        # Initialize counters for metrics
        self.total_crime_rec = self.total_crime_prec = 0
        self.total_time_score = self.total_amount_score = 0
        self.total_penalcode_index_rec = self.total_penalcode_index_prec = 0
        self.time_num = self.amount_num = 0

        # 以exp_data的键为主，过滤并对齐gen_data的键
        self.gen_data = {k: self.gen_data.get(k) for k in self.exp_data.keys()}
        assert self.gen_data.keys() == self.exp_data.keys(), "Mismatch between gen_data and exp_data keys after alignment"
        self.n = len(self.exp_data)  # Total number of items in data

    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = [json.loads(line) for line in file]
            if "gen_ans" in data[0]:
                return {item['input']: item['gen_ans'] for item in data}
            else:
                return {item['Fact'] : item for _, item in enumerate(data)}
            # if file_path.endswith('.json'):
            #     data = json.load(file)
            #     return {item['line_number']: item['gen_ans'] for item in data}
            # else:
            #     data = [json.loads(line) for line in file]
            #     return { i+1: item for i,item in enumerate(data)}

    def get_all_from_text(self, text):
        return get_crime(text), calc_time_sum(text), calc_amt_sum(text), get_penalcode_index_from_text(text)

    def calculate_recall_and_precision(self, expected, actual):
        expected_set = set(expected)
        actual_set = set(actual)
        true_positive = len(expected_set & actual_set)

        recall = true_positive / len(expected_set) if len(expected_set) > 0 else 0
        precision = true_positive / len(actual_set) if len(actual_set) > 0 else 0

        return recall, precision

    def calculate_recall_and_precision_article(self, expected, actual):
        expected_set = set(expected)
        actual_set = set(actual)
        true_set = (expected_set & actual_set)
        true_positive = len(expected_set & actual_set)
        actual_set = true_set

        recall = true_positive / len(expected_set) if len(expected_set) > 0 else 0
        precision = true_positive / len(actual_set) if len(actual_set) > 0 else 0

        return recall, precision

    def calculate_percent_for_judge(self, exp_val, act_val):
        if exp_val == act_val == 0:
            return 1.0
        if (exp_val >= 0 and act_val) < 0 or (exp_val < 0 and act_val >= 0):  # Different signs
            return 0.0
        if (exp_val - 10000) * (act_val - 10000) < 0:  # Both must either have or lack the death penalty
            return 0.0
        x = abs(exp_val - act_val) / max(exp_val, act_val)
        y = 1 - x
        return y

    def calc_metrics(self):
        output_dir = os.path.dirname(self.gen_file)
        eval_results_path = os.path.join(output_dir, 'evaluation_results.txt')
        no_pred_samples_path = os.path.join(output_dir, 'no_prediction_samples.txt')
        
        with open(eval_results_path, 'w', encoding='utf-8') as f, \
             open(no_pred_samples_path, 'w', encoding='utf-8') as no_pred_f:
            
            # 写入文件头
            f.write("===== 新评估会话 =====\n")
            no_pred_f.write("===== 无预测结果样本 =====\n")
            
            # 统计无预测结果的样本数量
            no_prediction_count = 0
            
            for exp_id, exp_ans in self.exp_data.items():
                gen_ans = self.gen_data[exp_id]
                gen_crime, gen_time, _, gen_penalcode_index = self.get_all_from_text(gen_ans)
                
                # 检查是否是无预测结果的样本
                is_no_prediction = (
                    len(gen_crime) == 0 and 
                    len(gen_penalcode_index) == 0 and 
                    gen_time == -1
                )
                
                exp_crime =[]
                exp_penalcode_index = []
                exp_time = 0 
                relevant_articles = exp_ans['relevant_articles']
                #相关法条的
                for item in relevant_articles:
                    match = re.search(r'\d+', item)
                    if match:
                        exp_penalcode_index.append(int(match.group()))
                ##多被告的
                for defendant in exp_ans['outcomes']:
                    for judgment in defendant['judgment']:
                            exp_crime.append(judgment['standard_accusation'])
                            # 添加错误处理
                            for item in judgment['article']:
                                match = re.search(r'\d+', item)
                                if match:
                                    exp_penalcode_index.append(int(match.group()))
                            exp_time += judgment['penalty']['imprisonment']
                exp_crime = list(set(exp_crime))
                exp_penalcode_index = list(set(exp_penalcode_index))
                crime_rec, crime_prec = self.calculate_recall_and_precision(exp_crime, gen_crime)
                penalcode_index_rec, penalcode_index_prec = self.calculate_recall_and_precision(exp_penalcode_index, gen_penalcode_index)

                # 累加结果
                self.total_crime_rec += crime_rec
                self.total_crime_prec += crime_prec
                self.total_penalcode_index_rec += penalcode_index_rec
                self.total_penalcode_index_prec += penalcode_index_prec

                time_score = -1  # 初始化时间分数
                if exp_time >= 0 or gen_time >= 0:
                    time_score = self.calculate_percent_for_judge(exp_time, gen_time)
                    self.total_time_score += time_score
                    self.time_num += 1

                # 写入评估结果
                result_text = (
                    f"样本ID: {exp_id}\n"
                    f"罪名真实值: {exp_crime}\t罪名预测值: {gen_crime}\n"
                    f"法条真实值: {exp_penalcode_index}\t法条预测值: {gen_penalcode_index}\n"
                    f"刑期真实值: {exp_time}\t刑期预测值: {gen_time}\n"
                    f"罪名召回率: {crime_rec:.4f}\t罪名精确率: {crime_prec:.4f}\n"
                    f"法条召回率: {penalcode_index_rec:.4f}\t法条精确率: {penalcode_index_prec:.4f}\n"
                    f"刑期得分: {time_score if time_score != -1 else 'N/A'}\t\n"
                    f"------------------------\n"
                )
                
                f.write(result_text)
                
                # 如果是无预测结果的样本，写入专门的文件
                if is_no_prediction:
                    no_prediction_count += 1
                    no_pred_f.write(f"\n样本 #{no_prediction_count}\n")
                    no_pred_f.write(f"原文: {exp_id}\n")
                    no_pred_f.write(f"生成结果: {gen_ans}\n")
                    no_pred_f.write(result_text)
            
            # 在无预测结果文件末尾写入统计信息
            no_pred_f.write(f"\n总计发现 {no_prediction_count} 个无预测结果样本\n")
            no_pred_f.write(f"占总样本比例: {(no_prediction_count/self.n)*100:.2f}%\n")
                
    def print_results(self):
        avg_crime_rec = self.total_crime_rec / self.n
        avg_crime_prec = self.total_crime_prec / self.n
        avg_penalcode_index_rec = self.total_penalcode_index_rec / self.n
        avg_penalcode_index_prec = self.total_penalcode_index_prec / self.n

        # Calculate F1 scores
        f1_crime = 2 * (avg_crime_prec * avg_crime_rec) / (avg_crime_prec + avg_crime_rec) if (avg_crime_prec + avg_crime_rec) != 0 else 0
        f1_penalcode_index = 2 * (avg_penalcode_index_prec * avg_penalcode_index_rec) / (avg_penalcode_index_prec + avg_penalcode_index_rec) if (avg_penalcode_index_prec + avg_penalcode_index_rec) != 0 else 0

        # Calculate average judge time score and average amount score
        avg_time_score = self.total_time_score / self.time_num if self.time_num > 0 else 0
        # avg_amount_score = self.total_amount_score / self.amount_num if self.amount_num > 0 else 0

        # Print the results
        print(f"Average Judge Time Score: {avg_time_score:.4f}")
        print(f"Average Crime Recall: {avg_crime_rec:.4f}, Average Crime Precision: {avg_crime_prec:.4f}, F1 Score: {f1_crime:.4f}")
        print(f"Average Penalcode Index Recall: {avg_penalcode_index_rec:.4f}, Average Penalcode Index Precision: {avg_penalcode_index_prec:.4f}, F1 Score: {f1_penalcode_index:.4f}")
        print(self.time_num)


def main():
    parser = argparse.ArgumentParser(description="Process a JSON file to calculate metrics.")
    parser.add_argument('--gen_file', type=str, default=r'data\多被告\cmdl\v30324\output.jsonl', help='Path to the input generated JSON file')
    parser.add_argument('--exp_file', type=str, default=r'data\多被告\cmdl\final.jsonl', help='Path to the expected JSON file')
    args = parser.parse_args()

    # Create an instance of MetricsCalculator
    calculator = MetricsCalculator(args.gen_file, args.exp_file)
    
    # Calculate the metrics
    calculator.calc_metrics()
    
    # Print the results
    calculator.print_results()
    print(f"This is the metrics from file {args.gen_file}!")


if __name__ == "__main__":
    main()
