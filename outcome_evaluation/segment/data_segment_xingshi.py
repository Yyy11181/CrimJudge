# -*- coding: utf-8 -*-
import os
import json
import time
import argparse
# 在文件开头添加当前目录到Python路径
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 然后正常导入
from xingshi_yishenpanjue_gongsu import DataSegmentXingshiGongsuYishenPanjue


class DataSegmentXingshi():
    
    def __init__(self, punctuation_replace=False):
        self.yishengongsu = DataSegmentXingshiGongsuYishenPanjue(punctuation_replace)

    def parse(self, wenshu):
        wenshu = self.yishengongsu.parse(wenshu)
        return wenshu
