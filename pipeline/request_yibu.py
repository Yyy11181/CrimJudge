import json
import requests
import time
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
import argparse
from typing import Dict, Any, Optional, List
import hashlib
import os
import pickle

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run legal document generation with LLM API.')
    parser.add_argument('--api_type', type=str, default="custom", #记得改
                       choices = ['openai', 'zhipu', 'siliconflow'], 
                       help='Type of API to use')
    parser.add_argument('--api_url', type=str, default= '',
                       help='Custom API URL (required for custom type)')
    parser.add_argument('--api_key', type=str,  #记得改
                       default="",  
                       help='API key for the service')  
    parser.add_argument('--model_name', type=str, 
                       default = 'gpt-4.1',  
                       help='Model name to use')
    parser.add_argument('--dataset_path', type=str, 
                       default=r"data\单人单罪\judge\gpt4.1_min.jsonl", 
                       help='Path to the dataset')
    parser.add_argument('--output_path', type=str, 
                       default=r"data\单人多罪\judge\gpt4.1_min11.jsonl", 
                       help='Path to save the output')
    parser.add_argument('--max_retries', type=int, default=3,
                       help='Maximum number of retries for API calls')
    parser.add_argument('--retry_delay', type=float, default=10,
                       help='Delay between retries in seconds')
    # 新增性能优化参数
    parser.add_argument('--max_workers', type=int, default=1,
                       help='Maximum number of concurrent workers')
    parser.add_argument('--batch_size', type=int, default=5,
                       help='Batch size for processing')
    parser.add_argument('--use_cache', default=False,
                       help='Enable caching of responses')
    parser.add_argument('--cache_dir', type=str, default=r"data\单人多罪\judge\O3\cache",
                       help='Directory for caching responses')
    parser.add_argument('--rate_limit', type=float, default=5,  # 增加速率限制参数
                       help='Minimum delay between requests (seconds)')
    parser.add_argument('--timeout', type=int, default=700,  # 增加默认超时时间
                       help='Request timeout in seconds')
    parser.add_argument('--use_async', default=True,
                       help='Use async HTTP requests (faster)')
    return parser.parse_args()

class CacheManager:
    def __init__(self, cache_dir: str, enabled: bool = True):
        self.cache_dir = cache_dir
        self.enabled = enabled
        if enabled and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[str]:
        """从缓存获取结果"""
        if not self.enabled:
            return None
        
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None
    
    def set(self, key: str, value: str):
        """设置缓存"""
        if not self.enabled:
            return
        
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception:
            pass

class AsyncLLMAPIClient:
    def __init__(self, api_type: str, api_key: str, model_name: str, 
                 api_url: str, max_retries: int = 3, 
                 retry_delay: float = 1.0, timeout: int = 30):
        self.api_type = api_type
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        
        if api_type == 'zhipu':
            self.api_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        elif api_type == 'siliconflow':
            self.api_url = "https://api.siliconflow.cn/v1/chat/completions"
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        elif api_type == 'custom':# 自定义API
            self.api_url = api_url
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        else:
            raise ValueError(f"Unsupported API type: {api_type}")
    
    def _prepare_payload(self, messages: list) -> Dict[str, Any]:
        """准备API请求的payload"""
        if self.api_type == 'custom' or self.api_type == 'siliconflow' or self.api_type == 'zhipu':
            return {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": 8192,
                "temperature": 0.7
                # "enable_thinking": False #专门是用于qwen235的
            }
    
    def _extract_response(self, response_data: Dict[str, Any]) -> str:
        """从API响应中提取生成的文本"""
        try:
            if self.api_type == 'custom' or self.api_type =='siliconflow' or self.api_type == 'zhipu':
                return response_data['choices'][0]['message']['content']
        except (KeyError, IndexError) as e:
            raise ValueError(f"Unexpected response format: {e}")
    
    async def generate_text_async(self, session: aiohttp.ClientSession, messages: list) -> str:
        payload = self._prepare_payload(messages)
        
        for attempt in range(self.max_retries):
            try:
                async with session.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(
                        total=self.timeout,
                        connect=30,
                        sock_read=self.timeout
                    ),
                    proxy=None  # 禁用代理
                ) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        return self._extract_response(response_data)
                    elif response.status in [429, 502, 503, 504]:  # 添加更多需要重试的状态码
                        wait_time = self.retry_delay * (2 ** attempt)  # 指数退避
                        print(f"服务器返回状态码 {response.status}，等待 {wait_time} 秒后重试...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        response_text = await response.text()
                        print(f"Error response: {response_text}")
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.retry_delay)
                            continue
                        raise Exception(f"API请求失败，状态码: {response.status}")
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                print(f"网络错误 (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    continue
                raise Exception(f"请求超时或网络错误: {str(e)}")
            
        raise Exception("达到最大重试次数")

    def generate_text_sync(self, messages: list) -> str:
        """同步生成文本（备用方法）"""
        payload = self._prepare_payload(messages)
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    return self._extract_response(response_data)
                elif response.status_code == 429:  # Rate limit
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        raise Exception(f"API request failed with status {response.status_code}: {response.text}")
                        
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise Exception(f"Request failed after {self.max_retries} attempts: {e}")
        
        raise Exception("Failed to get response from API")

def generate_reasoning_with_cache(fact: str, llm_client: AsyncLLMAPIClient, 
                                cache_manager: CacheManager) -> str:
    """使用缓存的生成推理"""
    # 检查缓存
    cache_key = cache_manager._get_cache_key(fact)
    cached_result = cache_manager.get(cache_key)
    if cached_result:
        return cached_result
    
    # 生成内容
    # prompt_template = open(r'utils\prompt.txt', 'r', encoding='utf-8').read()
    # input_content = prompt_template.format(fact=fact)

    input_content = f"""
【学术研究声明】
警告：以下内容为用于严肃法律人工智能学术研究的真实刑事案件描述，可能包含令人不适的暴力或犯罪细节。本次任务的目标是模拟司法裁判文书的生成，完全不旨在宣扬、模仿或鼓励任何有害行为。请在纯粹的学术研究框架下处理这些信息。”

案件事实：{fact}
请根据上面提供的事实描述，生成一篇完整且具有法律效力的中文的刑事判决书。生成的文书必须结构严谨、逻辑清晰，并且要包含以下个部分：

1. 开头
2. **事实描述**  
\- 请直接复述以下提供的事实描述，不得进行删减或改动：

3. **司法理由**  
\- 根据事实和描述，结合相关刑法条款、法律原则和司法解释，详细论述案件的法律分析，以"本院认为"开头。  
\- 分析内容应包括：  
​     \- 对证据的评估  
​     \- 犯罪构成要件的论证  
​     \- 相关法律条文的引用及其适用说明
     \- 对量刑结果的论证  
\- 请确保推理过程严谨、论证充分，为判决结果提供充分法律依据。

4. **判决结果**  
\- 在此部分明确给出法院的最终判决，以"判决书"开头。  
\- 判决内容应具体包括处罚措施（如刑期、罚金、附加刑等）及其法律依据，确保与前述司法理由相呼应，文书整体逻辑连贯。

**注意：**  确保文书所有部分均符合真实司法文书的写作规范，语言应正式、客观、清晰。
请参考以上示例，根据案件事实生成一份刑事判决书，结构完整，需包含案件事实陈述、法律分析、裁判理由及裁判结论等部分。
"""
    
    messages = [
        # {"role": "system", "content": "你是一个法律助理，提供帮助。"},
        {"role": "user", "content": input_content}
    ]
    
    # 使用同步方法生成（在线程池中）
    response = llm_client.generate_text_sync(messages)
    
    # 缓存结果
    cache_manager.set(cache_key, response)
    
    return response

async def process_batch_async(batch_data: List[Dict], llm_client: AsyncLLMAPIClient, 
                            cache_manager: CacheManager, rate_limit: float) -> List[Dict]:
    """异步处理批量数据"""
    results = []
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for item in batch_data:
            # 检查缓存，如果相同的Fact之前已经请求过，则直接使用缓存结果
            cache_key = cache_manager._get_cache_key(item['Fact'])
            cached_result = cache_manager.get(cache_key) #根据缓存键获取缓存结果
            
            if cached_result:
                result = {
                    "line_number": item.get('line_number', 0),
                    "input": item['Fact'],
                    "gen_ans": cached_result,
                    # "exp_ans": item.get('Full Document', ''),
                    "from_cache": True
                }
                results.append(result)
            else:
                # 创建异步任务
                task = process_single_item_async(session, item, llm_client, cache_manager)
                tasks.append(task)
                
                # 添加速率限制
                if rate_limit > 0:
                    await asyncio.sleep(rate_limit)
        
        # 等待所有任务完成
        if tasks:
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in completed_results:
                if not isinstance(result, Exception):
                    results.append(result)
                else:
                    print(f"Task failed: {result}")
    
    return results

async def process_single_item_async(session: aiohttp.ClientSession, item: Dict, 
                                  llm_client: AsyncLLMAPIClient, cache_manager: CacheManager) -> Dict:
    """异步处理单个项目"""

    prompt_template = open(r'utils\prompt.txt', 'r', encoding='utf-8').read()
    input_content = prompt_template.format(fact=item['Fact'])

#     input_content = f"""
# 【学术研究声明】
# 警告：以下内容为用于严肃法律人工智能学术研究的真实刑事案件描述，可能包含令人不适的暴力或犯罪细节。本次任务的目标是模拟司法裁判文书的生成，完全不旨在宣扬、模仿或鼓励任何有害行为。请在纯粹的学术研究框架下处理这些信息。”

# 案件事实：{item['Fact']}
# 请根据上面提供的事实描述，生成一篇完整且具有法律效力的中文的刑事判决书。生成的文书必须结构严谨、逻辑清晰，并且要包含以下个部分：

# 1. 开头
# 2. **事实描述**  
# \- 请直接复述以下提供的事实描述，不得进行删减或改动：

# 3. **司法理由**  
# \- 根据事实和描述，结合相关刑法条款、法律原则和司法解释，详细论述案件的法律分析，以"本院认为"开头。  
# \- 分析内容应包括：  
# ​     \- 对证据的评估  
# ​     \- 犯罪构成要件的论证  
# ​     \- 相关法律条文的引用及其适用说明  
#      \- 对量刑结果的论证
# \- 请确保推理过程严谨、论证充分，为判决结果提供充分法律依据。

# 4. **判决结果**  
# \- 在此部分明确给出法院的最终判决，以"判决书"开头。  
# \- 判决内容应具体包括处罚措施（如刑期、罚金、附加刑等）及其法律依据，确保与前述司法理由相呼应，文书整体逻辑连贯。

# **注意：**  确保文书所有部分均符合真实司法文书的写作规范，语言应正式、客观、清晰。判决书的格式可以参考示例中的格式。
# 请参考以上示例，根据案件事实生成一份刑事判决书，结构完整，需包含案件事实陈述、法律分析、裁判理由及裁判结论等部分。

# """
    
    messages = [
        # {"role": "system", "content": "你是一个法律助理，提供帮助。"},
        {"role": "user", "content": input_content}
    ]
    
    try:
        gen_ans = await llm_client.generate_text_async(session, messages)
        
        # 缓存结果
        cache_key = cache_manager._get_cache_key(item['Fact'])#根据Fact生成缓存键
        cache_manager.set(cache_key, gen_ans) #将结果存入缓存
        
        return {
            "line_number": item.get('line_number', 0),
            "input": item['Fact'],
            "gen_ans": gen_ans,
            # "exp_ans": item.get('Full Document', ''),
            "from_cache": False
        }
    except Exception as e:
        print(f"Error processing item: {e}")
        return {
            "line_number": item.get('line_number', 0),
            "input": item['Fact'],
            "gen_ans": f"Error: {str(e)}",
            # "exp_ans": item.get('Full Document', ''),
            "from_cache": False
        }

def process_with_threading(data_list: List[Dict], llm_client: AsyncLLMAPIClient, 
                         cache_manager: CacheManager, max_workers: int, rate_limit: float) -> List[Dict]:
    """使用线程池处理数据"""
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        futures = []
        for item in data_list:
            future = executor.submit(generate_reasoning_with_cache, item['Fact'], llm_client, cache_manager)
            futures.append((future, item))
            
            # 速率限制
            if rate_limit > 0:
                time.sleep(rate_limit)
        
        # 获取结果
        for future, item in tqdm(futures, desc="Processing"):
            try:
                gen_ans = future.result()
                result = {
                    "line_number": item.get('line_number', 0),
                    "input": item['Fact'],
                    "gen_ans": gen_ans,
                    # "exp_ans": item.get('Full Document', '')
                }
                results.append(result)
            except Exception as e:
                print(f"Error processing item: {e}")
                continue
    
    return results

def process_dataset_optimized(dataset_path: str, output_path: str, llm_client: AsyncLLMAPIClient, 
                            cache_manager: CacheManager, args):
    """优化的数据集处理"""
    # 读取所有数据
    data_list = []
    with open(dataset_path, "r", encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1): #从第一行开始计算id
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                item['line_number'] = line_num  
                # item['line_number'] = item["CaseId"]  #后面需要把这个换成id,为了和原始的文件对齐。item["id"]
                data_list.append(item)
            except json.JSONDecodeError as e:
                print(f"JSON decode error on line {line_num}: {e}")
                continue
    
    print(f"Total items to process: {len(data_list)}")
    
    # 分批处理！！
    data_list =  data_list # 限制处理前100条数据，便于测试和调试
    all_results = []
    batch_size = args.batch_size
    
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1} ({len(batch)} items)")
        
        if args.use_async:
            # 使用异步处理
            results = asyncio.run(process_batch_async(batch, llm_client, cache_manager, args.rate_limit))
        else:
            # 使用线程池处理
            results = process_with_threading(batch, llm_client, cache_manager, args.max_workers, args.rate_limit)
        
        all_results.extend(results)
        
        # 中间保存
        if i % (batch_size * 5) == 0:  # 每5个批次保存一次
            temp_output = output_path.replace('.json', f'_temp_{i}.json')
            with open(temp_output, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"Intermediate results saved to {temp_output}")
    
    # 最终保存
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
        # json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"Final results saved to {output_path}")
    print(f"Total processed: {len(all_results)} items")
    
    # 统计信息
    cached_count = sum(1 for r in all_results if r.get('from_cache', False))
    print(f"Cache hits: {cached_count}/{len(all_results)} ({cached_count/len(all_results)*100:.1f}%)")

def main():
    args = parse_arguments()
    print(f"启动参数: {vars(args)}")
    
    # 检查数据文件
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset file not found: {args.dataset_path}")
        return
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 初始化缓存管理器
    cache_manager = CacheManager(args.cache_dir, args.use_cache)
    
    # 创建LLM客户端
    llm_client = AsyncLLMAPIClient(
        api_type=args.api_type,
        api_key=args.api_key,  # 使用命令行参数中的api_key
        model_name=args.model_name,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        timeout=args.timeout,
        api_url=args.api_url
    )
    
    # 处理数据集
    start_time = time.time()
    process_dataset_optimized(args.dataset_path, args.output_path, llm_client, cache_manager, args)
    end_time = time.time()
    
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
