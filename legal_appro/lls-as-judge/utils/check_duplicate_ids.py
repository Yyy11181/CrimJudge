import pandas as pd
import os

def check_duplicate_ids():
    """
    检查三个打分文件中是否存在重复的id
    """
    # 定义文件路径
    files = [
    ]
    
    print("开始检查重复id...")
    
    for file in files:
        if os.path.exists(file):
            try:
                df = pd.read_excel(file)
                
                if 'id' in df.columns:
                    # 检查id列是否有重复值
                    duplicate_ids = df[df.duplicated('id', keep=False)]['id'].unique()
                    
                    if len(duplicate_ids) > 0:
                        print(f"\n文件 {file} 中存在重复的id:")
                        print(f"重复的id列表: {sorted(duplicate_ids)}")
                        print(f"重复的id数量: {len(duplicate_ids)}")
                        
                        # 显示每个重复id出现的次数
                        id_counts = df['id'].value_counts()
                        for id_val in sorted(duplicate_ids):
                            count = id_counts[id_val]
                            print(f"id {id_val} 出现了 {count} 次")
                    else:
                        print(f"\n文件 {file} 中没有重复的id")
                else:
                    print(f"\n文件 {file} 中没有id列")
            except Exception as e:
                print(f"读取文件 {file} 时出错: {e}")
        else:
            print(f"文件 {file} 不存在")
    
    print("\n检查完成！")

if __name__ == "__main__":
    check_duplicate_ids()