#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将JSON文件转换为Excel文件的工具，支持命令行参数
"""

import json
import pandas as pd
import os
import argparse
from datetime import datetime


def convert_json_to_excel(json_file_path, output_excel_path=None):
    """
    将JSON文件转换为Excel文件
    
    参数:
        json_file_path: JSON文件路径
        output_excel_path: 输出Excel文件路径，默认为json_file_path.xlsx
    """
    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"成功读取JSON文件，共包含 {len(data)} 条记录")
        
        # 如果没有指定输出路径，默认使用JSON文件名加上.xlsx后缀
        if output_excel_path is None:
            output_excel_path = os.path.splitext(json_file_path)[0] + '.xlsx'
        
        # 将JSON数据转换为DataFrame
        # 使用pd.DataFrame而不是pd.json_normalize，这样不会展开嵌套的字典
        df = pd.DataFrame(data)
        
        # 将所有字典类型的字段转换为标准JSON字符串
        for col in df.columns:
            # 检查该列是否包含字典类型的值
            if df[col].apply(lambda x: isinstance(x, dict)).any():
                # 将字典转换为JSON字符串
                df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else x)
        
        print(f"数据转换为DataFrame成功，包含 {len(df.columns)} 列")
        
        # 保存为Excel文件
        with pd.ExcelWriter(output_excel_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='matching_data')
            
            # 设置工作表格式
            workbook = writer.book
            worksheet = writer.sheets['matching_data']
            
            # 自动调整列宽
            for i, col in enumerate(df.columns):
                # 计算列内容的最大宽度
                max_width = max(df[col].astype(str).apply(len).max(), len(col))
                # 设置列宽，增加一点缓冲
                worksheet.set_column(i, i, min(max_width + 2, 50))  # 最大宽度限制为50
        
        print(f"Excel文件创建成功: {output_excel_path}")
        print(f"文件大小: {os.path.getsize(output_excel_path):,} 字节")
        return output_excel_path
        
    except json.JSONDecodeError as e:
        print(f"JSON格式错误: {e}")
        return None
    except Exception as e:
        print(f"转换过程中出错: {e}")
        return None


def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='将JSON文件转换为Excel文件')
    parser.add_argument('--json_file', type=str, default='matching_data_new.json',
                      help='要转换的JSON文件路径')
    parser.add_argument('--output', type=str, default=None,
                      help='输出Excel文件路径')
    
    # 解析命令行参数
    args = parser.parse_args()
    json_file = args.json_file
    output_excel = args.output
    
    # 检查文件是否存在
    if not os.path.exists(json_file):
        print(f"文件不存在: {json_file}")
        return
    
    print(f"开始转换 {json_file} 到Excel...")
    
    # 执行转换
    output_file = convert_json_to_excel(json_file, output_excel)
    
    if output_file:
        print("转换完成！")
        print(f"输出文件: {output_file}")
    else:
        print("转换失败！")


if __name__ == "__main__":
    main()
