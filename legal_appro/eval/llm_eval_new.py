import json
import os
import httpx
from tqdm import tqdm
import concurrent.futures
import argparse
import time
import asyncio
import aiohttp
from typing import List, Dict
import logging
from dataclasses import dataclass



# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """API配置类"""
    max_workers: int = 60  # 增加并发数
    max_retries: int = 3   # 增加重试次数，提高成功率
    timeout: int = 360      # 请求超时时间
    rate_limit: float = 0.5  # 请求间隔（秒）
    batch_size: int = 50   # 批处理大小


class AsyncOpenAI_API:
    """异步OpenAI API客户端"""
    def __init__(self, api_key: str, config: APIConfig):
        self.api_key = api_key
        self.config = config
        self.base_url = ""
        # 配置连接池参数以提高性能
        self.connector = aiohttp.TCPConnector(
            limit=100,  # 总连接池大小
            limit_per_host=50,  # 每个主机的连接数
            keepalive_timeout=60,  # 保持连接时间
            enable_cleanup_closed=True
        )
        
    async def create_session(self):
        """创建HTTP会话"""
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        return aiohttp.ClientSession(
            connector=self.connector,
            timeout=timeout,
            headers={'Authorization': f'Bearer {self.api_key}'}
        )
    
    async def chat_async(self, session: aiohttp.ClientSession, query: str) -> dict:
        """异步聊天请求"""
        url = f"{self.base_url}/v1/chat/completions" if self.base_url else "https://api.openai.com/v1/chat/completions"
        
        payload = {
            "model": "Pro/deepseek-ai/DeepSeek-R1",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ],
            "stream": False,
            "temperature": 0,
            "top_p": 0
        }
        
        async with session.post(url, json=payload) as response:
            # 如果状态码不是200-299，aiohttp的raise_for_status()会抛出ClientResponseError
            response.raise_for_status() 
            # 如果状态码正常，解码JSON并返回
            try:
                return await response.json()
            except json.JSONDecodeError as json_error:
                # 如果返回的不是合法的JSON
                raw_text = await response.text()
                logger.error(f"无法解析JSON，原始文本: {raw_text}")
                raise json_error # 重新抛出异常
        
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                logger.info(f"收到200 OK响应，内容为: {json.dumps(result, indent=2, ensure_ascii=False)}")
                 # 在打印之后再尝试解析
                if 'choices' in result and result['choices']:
                    return result['choices'][0]['message']['content']
                else:
                    # 如果没有choices，说明是API层面的错误，抛出异常
                    raise Exception(f"API返回200 OK但内容有误: {result.get('error')}")
                # return result['choices'][0]['message']['content']
            else:
                error_text = await response.text()
                raise Exception(f"API请求失败: {response.status}, {error_text}")


class SyncOpenAI_API:
    """同步OpenAI API客户端（优化版）"""
    def __init__(self, api_key: str, config: APIConfig):
        self.api_key = api_key
        self.config = config
        self.base_url = ""
        
        # 使用httpx客户端替代OpenAI客户端以获得更好的性能控制
        self.client = httpx.Client(
            timeout=config.timeout,
            limits=httpx.Limits(
                max_keepalive_connections=50,
                max_connections=100,
                keepalive_expiry=60
            )
        )
        
    def chat(self, query: str) -> dict:
        """同步聊天请求"""
        url = f"{self.base_url}/v1/chat/completions" if self.base_url else "https://api.openai.com/v1/chat/completions"
        
        payload = {
            "model": "Pro/deepseek-ai/DeepSeek-R1",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ],
            "stream": False,
            "temperature": 0,
            "top_p": 0
        }
        
        headers = {'Authorization': f'Bearer {self.api_key}'}
        
        response = self.client.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            raise Exception(f"API请求失败: {response.status_code}, {response.text}")
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'client'):
            self.client.close()


class OptimizedChatClient:
    """优化的聊天客户端"""
    def __init__(self, api_key: str, config: APIConfig = None, use_async: bool = True):
        self.api_key = api_key
        self.config = config or APIConfig()
        self.use_async = use_async
        
        if use_async:
            self.client = AsyncOpenAI_API(api_key, self.config)
        else:
            self.client = SyncOpenAI_API(api_key, self.config)
    
    async def chat_async(self, session, query: str) -> dict:
        """异步聊天"""
        # 如果使用异步环境，则直接调用异步聊天方法
        if self.use_async:
            return await self.client.chat_async(session, query)
        else:
            # 在异步环境中运行同步代码
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.client.chat, query)
    
    def chat_sync(self, query: str) -> dict:
        """同步聊天"""
        return self.client.chat(query)


async def ask_prompt_async(json_object: Dict, client: OptimizedChatClient, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore) -> Dict:
    """
    经过优化的异步处理函数，能区分网络错误和解析错误。
    """
    async with semaphore:
        prompt = json_object['prompt']
        attempts = 0

        while attempts < client.config.max_retries:
            try:
                # ------------------- 1. API 调用阶段 -------------------
                # 添加速率限制
                if client.config.rate_limit > 0 and attempts > 0:
                    await asyncio.sleep(client.config.rate_limit)

                # 调用客户端，假设 chat_async 现在会抛出非200状态的异常
                # 并且成功时返回的是解码后的JSON字典
                api_response_dict = await client.chat_async(session, prompt)

                # ------------------- 2. 解析阶段 -------------------
                # 在一个独立的 try-except 块中进行解析，这样可以捕获到解析错误
                try:
                    content = api_response_dict['choices'][0]['message']['content']
                    logger.info(f"成功处理并解析ID: {json_object['id']}")
                    return {'id': json_object['id'], 'response': content}

                except (KeyError, IndexError, TypeError) as parse_error:
                    # 这是解析错误！说明API返回的JSON结构不对
                    logger.error(
                        f"ID {json_object['id']} - 解析响应失败: {parse_error}. "
                        f"API返回的原始数据: {json.dumps(api_response_dict, ensure_ascii=False, indent=2)}"
                    )
                    # 这种错误不应该重试，直接返回失败
                    return {'id': json_object['id'], 'response': f"Error: 解析响应失败, 原始返回: {api_response_dict}"}

            # ------------------- 3. 异常处理阶段 -------------------
            except (aiohttp.ClientError, asyncio.TimeoutError) as network_error:
                # 这是网络错误或超时，应该重试
                attempts += 1
                logger.warning(f"ID {json_object['id']} - 网络请求失败 (第{attempts}次): {network_error}")
                if attempts < client.config.max_retries:
                    wait_time = min(2 ** attempts, 10) # 指数退避
                    await asyncio.sleep(wait_time)
                else:
                    # 达到最大重试次数
                    logger.error(f"ID {json_object['id']} - 网络错误超过最大尝试次数")
                    return {'id': json_object['id'], 'response': "Error: 网络错误超过最大尝试次数"}

            except Exception as e:
                # 捕获其他所有意想不到的异常
                attempts += 1
                logger.critical(f"ID {json_object['id']} - 出现未知严重错误 (第{attempts}次): {e}")
                if attempts < client.config.max_retries:
                    wait_time = min(2 ** attempts, 10)
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"ID {json_object['id']} - 未知错误超过最大尝试次数")
                    return {'id': json_object['id'], 'response': f"Error: 未知错误 - {e}"}

        # 如果循环正常结束（虽然理论上不会，因为总会 return），也返回错误
        return {'id': json_object['id'], 'response': "Error: 达到最大尝试次数"}

def ask_prompt_sync(json_object: Dict, client: OptimizedChatClient) -> Dict:
    """同步处理单个提示（用于线程池）"""
    prompt = json_object['prompt']
    attempts = 0
    
    while attempts < client.config.max_retries:
        try:
            # 添加速率限制
            if client.config.rate_limit > 0 and attempts > 0:
                time.sleep(client.config.rate_limit)

            # 调用客户端，现在返回的是解码后的JSON字典
            api_response_dict = client.chat_sync(prompt)

            # 解析响应
            try:
                content = api_response_dict['choices'][0]['message']['content']
                logger.info(f"成功处理并解析ID: {json_object['id']}")
                return {'id': json_object['id'], 'response': content}

            except (KeyError, IndexError, TypeError) as parse_error:
                # 解析错误！说明API返回的JSON结构不对
                logger.error(
                    f"ID {json_object['id']} - 解析响应失败: {parse_error}. "
                    f"API返回的原始数据: {json.dumps(api_response_dict, ensure_ascii=False, indent=2)}"
                )
                # 这种错误不应该重试，直接返回失败
                return {'id': json_object['id'], 'response': f"Error: 解析响应失败, 原始返回: {api_response_dict}"}

        except Exception as e:
            # 捕获其他所有异常
            attempts += 1
            logger.warning(f"ID {json_object['id']} 第{attempts}次尝试失败: {str(e)}")
            
            if attempts < client.config.max_retries:
                wait_time = min(2 ** attempts, 10)
                time.sleep(wait_time)
    
    logger.error(f"ID {json_object['id']} 超过最大尝试次数")
    return {'id': json_object['id'], 'response': "Error: 超过最大尝试次数"}


async def process_batch_async(json_data: List[Dict], processed_ids: set, api_key: str, config: APIConfig) -> List[Dict]:
    """异步批处理"""
    client = OptimizedChatClient(api_key, config, use_async=True)
    
    # 过滤已处理的数据
    unprocessed_data = [item for item in json_data if item['id'] not in processed_ids]
    
    if not unprocessed_data:
        logger.info("没有需要处理的数据")
        return []
    
    # 创建信号量限制并发数
    semaphore = asyncio.Semaphore(config.max_workers)
    
    async with await client.client.create_session() as session:
        # 创建所有任务
        tasks = [
            ask_prompt_async(json_object, client, session, semaphore)
            for json_object in unprocessed_data
        ]
        
        # 使用进度条
        results = []
        with tqdm(total=len(tasks), desc="异步处理中") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)
    
    return results


def process_batch_sync(json_data: List[Dict], processed_ids: set, api_key: str, config: APIConfig) -> List[Dict]:
    """同步批处理（使用线程池）"""
    client = OptimizedChatClient(api_key, config, use_async=False)
    
    # 过滤已处理的数据
    unprocessed_data = [item for item in json_data if item['id'] not in processed_ids]
    
    if not unprocessed_data:
        logger.info("没有需要处理的数据")
        return []
    
    results = []
    
    with tqdm(total=len(unprocessed_data), desc="同步处理中") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            # 提交所有任务
            future_to_data = {
                executor.submit(ask_prompt_sync, json_object, client): json_object
                for json_object in unprocessed_data
            }
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_data):
                result = future.result()
                results.append(result)
                pbar.update(1)
    
    return results


def save_results_incrementally(results: List[Dict], output_file_path: str):
    """增量保存结果"""
    with open(output_file_path, 'a', encoding='utf-8') as out_file:
        for result in results:
            out_file.write(json.dumps(result, ensure_ascii=False) + '\n')


def ask_file_optimized(input_file_path: str, output_file_path: str, api_key: str, 
                      config: APIConfig = None, use_async: bool = True):
    """优化的文件处理函数"""
    config = config or APIConfig()
    
    # 读取已处理的ID
    processed_ids = set()
    if os.path.exists(output_file_path): #如果存在输出文件，读取已处理的ID
        with open(output_file_path, 'r', encoding='utf-8') as out_file:
            for line in out_file:
                try:
                    result = json.loads(line)
                    processed_ids.add(result['id'])
                except json.JSONDecodeError:
                    continue
    
    logger.info(f"已处理 {len(processed_ids)} 个项目")
    
    # 读取输入数据
    with open(input_file_path, 'r', encoding='utf-8') as file:
        json_data = [json.loads(line) for line in file.readlines()]#这边可以进行测试
    
    logger.info(f"总项目数: {len(json_data)}")
    
    # 分批处理以避免内存问题
    batch_size = config.batch_size
    all_results = []
    
    for i in range(0, len(json_data), batch_size):
        batch = json_data[i:i + batch_size]
        logger.info(f"处理批次 {i//batch_size + 1}/{(len(json_data) + batch_size - 1)//batch_size}")
        
        if use_async:
            # 异步处理
            results = asyncio.run(process_batch_async(batch, processed_ids, api_key, config))
        else:
            # 同步处理
            results = process_batch_sync(batch, processed_ids, api_key, config)
        
        # 增量保存
        if results:
            save_results_incrementally(results, output_file_path)
            all_results.extend(results)
            # 更新已处理ID集合
            processed_ids.update(result['id'] for result in results)
    
    logger.info(f"本次处理了 {len(all_results)} 个结构体")
    
    # 按ID排序并重写文件
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r', encoding='utf-8') as out_file:
            all_saved_results = []
            for line in out_file:
                try:
                    all_saved_results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        sorted_results = sorted(all_saved_results, key=lambda x: x['id'])
        
        with open(output_file_path, 'w', encoding='utf-8') as out_file:
            for result in sorted_results:
                out_file.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        logger.info("所有结构体按ID排序写入完成")


def main():
    parser = argparse.ArgumentParser(description='优化的模型处理工具')
    parser.add_argument('--model_name', type=str, default='all', help='使用的模型名称')
    parser.add_argument('--api_key', type=str, default="", help='API密钥')
    parser.add_argument('--task_name', type=str, default='criminal_new', 
                       help='要处理的任务名称 (defense, fact, reasoning, judgement, criminal)')
    parser.add_argument('--max_workers', type=int, default=100, help='最大并发数')
    parser.add_argument('--use_async', action='store_true', default=True, help='使用异步处理')
    parser.add_argument('--batch_size', type=int, default=50, help='批处理大小')
    parser.add_argument('--rate_limit', type=float, default=0.1, help='请求间隔（秒）')
    
    args = parser.parse_args()
    
    # 创建配置
    config = APIConfig(
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        rate_limit=args.rate_limit
    )
    
    # #定义路径
    # input_dir = f"eval/prompt/{args.model_name}"
    # output_dir = f"eval/llm_eval_cluade_SLJA/{args.model_name}"
    
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    
    # 定义任务
    tasks = ['defense', 'fact', 'reasoning', 'judgement', 'criminal_new']
    if args.task_name in tasks:
        tasks = [args.task_name]
    
    # 处理每个任务
    # model = args.model_name
    model_names = [name.strip() for name in args.model_name.split(',') if name.strip()]
    for model in model_names:
        input_dir = f"eval/prompt/{model}"
        output_dir = f"eval/llm_eval_r1_all/{model}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for task in tasks:
            logger.info(f"开始处理任务: {task} in {model}")
            input_file_path = os.path.join(input_dir, f"{task}_eval_prompt.json")
            output_file_path = os.path.join(output_dir, f"{task}_7.json")
            
            if not os.path.exists(input_file_path):
                logger.warning(f"输入文件不存在: {input_file_path}")
                continue
            
            start_time = time.time()
            ask_file_optimized(
                input_file_path, 
                output_file_path, 
                args.api_key, 
                config, 
                args.use_async
            )
            end_time = time.time()
            
            logger.info(f"任务 {task} 完成，耗时: {end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    main()

