import json
import time
import asyncio
import dashscope
from dashscope import Generation
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
import pickle
import argparse
from typing import Dict, Any, Optional, List
from tqdm import tqdm
#'farui-plus'
def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Run legal document generation with DashScope LLM API.')
    
    # --- API and Model Configuration ---
    parser.add_argument('--api_key', type=str,
                        default=os.environ.get("DASHSCOPE_API_KEY", ""),
                        help='API key for DashScope service. Can also be set via DASHSCOPE_API_KEY env var.')
    parser.add_argument('--model_name', type=str,
                        default='farui-plus',
                        help='DashScope model name to use (e.g., qwen-long, qwen-max, qwen-turbo)')

    # --- Data Paths ---
    parser.add_argument('--dataset_path', type=str,
                        default=r"data\单人多罪\judge\法睿\min.jsonl",
                        help='Path to the input dataset (JSONL format)')
    parser.add_argument('--output_path', type=str,
                        default=r"data\单人多罪\judge\法睿\minout.jsonl",
                        help='Path to save the output results')

    # --- Performance and Execution Control ---
    parser.add_argument('--max_workers', type=int, default=10,
                        help='Maximum number of concurrent workers')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Number of items to process in each batch')
    parser.add_argument('--timeout', type=int, default=120,
                        help='Request timeout in seconds for API calls')
    parser.add_argument('--max_retries', type=int, default=1,
                        help='Maximum number of retries for failed API calls')
    parser.add_argument('--retry_delay', type=float, default=5.0,
                        help='Initial delay between retries in seconds (will increase exponentially)')
    
    # --- Execution Mode ---
    parser.add_argument('--use_async', action='store_true', default=True,
                        help='Enable async processing mode (default)')
    parser.add_argument('--no-async', dest='use_async', action='store_false',
                        help='Disable async processing and use threading instead')

    # --- Caching ---
    parser.add_argument('--use_cache', action='store_true', default=False,
                        help='Enable caching of API responses (default)')
    parser.add_argument('--no-cache', dest='use_cache', action='store_false',
                        help='Disable caching of API responses')
    parser.add_argument('--cache_dir', type=str, default=r"data\单人多罪\judge\法睿\cache",
                        help='Directory to store cache files')

    return parser.parse_args()


class CacheManager:
    """管理API请求的缓存，避免重复请求"""
    def __init__(self, cache_dir: str, enabled: bool = True):
        self.cache_dir = cache_dir
        self.enabled = enabled
        if self.enabled and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            print(f"Cache directory created at: {self.cache_dir}")

    def _get_cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def get(self, text: str) -> Optional[str]:
        if not self.enabled:
            return None
        
        key = self._get_cache_key(text)
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except (pickle.UnpicklingError, EOFError) as e:
                print(f"Warning: Could not read cache file {cache_file}. It will be overwritten. Error: {e}")
                return None
        return None

    def set(self, text: str, value: str):
        if not self.enabled:
            return
        
        key = self._get_cache_key(text)
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except IOError as e:
            print(f"Error: Could not write to cache file {cache_file}. Error: {e}")


class DashScopeLLMClient:
    """使用DashScope SDK与大语言模型交互的客户端"""
    def __init__(self, api_key: str, model_name: str, max_retries: int, retry_delay: float, timeout: int):
        if not api_key:
            raise ValueError("DashScope API key is required.")
        dashscope.api_key = api_key
        
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        # 读取一次模板，避免在每次调用时都读取文件
        try:
            with open(r'utils\prompt.txt', 'r', encoding='utf-8') as f:
                self.prompt_template = f.read()
        except FileNotFoundError:
            print("Warning: utils\prompt.txt not found. Using a default template.")
            self.prompt_template = "请根据以下案情事实，生成一份完整的法律文书：\n\n案情事实：\n{fact}\n\n生成的文书："


    def _prepare_prompt(self, fact: str) -> str:
        """准备API请求的最终prompt字符串"""
        return self.prompt_template.format(fact=fact)



    def _extract_response(self, response) -> str:
        """从API响应中安全地提取文本内容"""
        try:
            # 检查响应状态
            if hasattr(response, 'status_code') and response.status_code == 200:
                # 对于farui-plus等模型，响应在output.text中
                if hasattr(response, 'output') and hasattr(response.output, 'text'):
                    return response.output.text
                # 对于使用messages格式的模型，响应在output.choices中
                elif hasattr(response, 'output') and hasattr(response.output, 'choices') and response.output.choices:
                    return response.output.choices[0].message.content
                # 直接的text属性
                elif hasattr(response, 'text'):
                    return response.text
                else:
                    raise ValueError(f"Cannot find text content in response structure")
            else:
                # 处理错误响应
                error_msg = getattr(response, 'message', 'Unknown error')
                request_id = getattr(response, 'request_id', 'Unknown')
                status = getattr(response, 'status_code', 'Unknown')
                raise RuntimeError(f"API request failed with status {status}: {error_msg} (Request ID: {request_id})")
        except (KeyError, IndexError, AttributeError) as e:
            raise ValueError(f"Failed to extract content from response. Error: {e}")

    def generate_text_sync(self, fact: str) -> str:
        """同步调用DashScope API生成文本，包含重试逻辑"""
        # 准备最终的prompt字符串
        final_prompt = self._prepare_prompt(fact)
        
        for attempt in range(self.max_retries):
            try:
                # *** FIX: Use 'prompt' argument instead of 'messages' ***
                response = Generation.call(
                    model=self.model_name,
                    prompt = final_prompt,
                    # result_format='message'
                )
                print(response)
                return self._extract_response(response)
            except Exception as e:
                print(f"Error on sync call (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Sync call failed after {self.max_retries} attempts for fact: {fact[:100]}...") from e
        raise Exception("Sync call failed after reaching max retries.")

    async def generate_text_async(self, fact: str) -> str:
        """异步调用DashScope API"""
        try:
            response_content = await asyncio.to_thread(
                self.generate_text_sync, fact
            )
            return response_content
        except Exception as e:
            raise e


async def process_item_async(item: Dict, llm_client: DashScopeLLMClient, cache_manager: CacheManager, rate_limiter: asyncio.Semaphore) -> Dict:
    """使用异步方法处理单个数据项"""
    fact = item['Fact']
    gen_ans = None
    from_cache = False

    async with rate_limiter:
        cached_result = cache_manager.get(fact)
        
        if cached_result:
            gen_ans = cached_result
            from_cache = True
        else:
            try:
                gen_ans = await llm_client.generate_text_async(fact)
                cache_manager.set(fact, gen_ans)
            except Exception as e:
                print(f"Failed to process item {item.get('line_number', 'N/A')}: {e}")
                gen_ans = f"Error: {e}"
            from_cache = False
            
        return {
            "line_number": item.get('line_number', 0),
            "input": fact,
            "gen_ans": gen_ans,
            "from_cache": from_cache
        }


def process_item_sync(item: Dict, llm_client: DashScopeLLMClient, cache_manager: CacheManager) -> Dict:
    """使用同步方法处理单个数据项（用于线程池）"""
    fact = item['Fact']
    cached_result = cache_manager.get(fact)
    
    if cached_result:
        gen_ans = cached_result
        from_cache = True
    else:
        try:
            gen_ans = llm_client.generate_text_sync(fact)
            cache_manager.set(fact, gen_ans)
        except Exception as e:
            print(f"Failed to process item {item.get('line_number', 'N/A')}: {e}")
            gen_ans = f"Error: {e}"
        from_cache = False
        
    return {
        "line_number": item.get('line_number', 0),
        "input": fact,
        "gen_ans": gen_ans,
        "from_cache": from_cache
    }


async def run_async_processing(data_list: List[Dict], llm_client: DashScopeLLMClient, cache_manager: CacheManager, args):
    """协调异步处理流程"""
    all_results = []
    rate_limiter = asyncio.Semaphore(args.max_workers) 
    
    pbar = tqdm(total=len(data_list), desc="Processing Async")
    
    for i in range(0, len(data_list), args.batch_size):
        batch_data = data_list[i:i + args.batch_size]
        tasks = [process_item_async(item, llm_client, cache_manager, rate_limiter) for item in batch_data]
        results = await asyncio.gather(*tasks)
        all_results.extend(results)
        pbar.update(len(batch_data))
        
        if i > 0 and (i // args.batch_size) % 5 == 0:
            save_results(all_results, args.output_path, is_temp=True, batch_num=(i // args.batch_size))
            
    pbar.close()
    return all_results

def run_sync_processing_with_threading(data_list: List[Dict], llm_client: DashScopeLLMClient, cache_manager: CacheManager, args):
    """使用线程池进行同步处理"""
    all_results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_item_sync, item, llm_client, cache_manager): item for item in data_list}
        for future in tqdm(as_completed(futures), total=len(data_list), desc="Processing Sync (Threading)"):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                print(f"Critical error in thread future: {e}")
            
    return all_results
    
def load_dataset(dataset_path: str) -> List[Dict]:
    """从JSONL文件加载数据集"""
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found: {dataset_path}")
        return []
        
    data_list = []
    with open(dataset_path, "r", encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                item['line_number'] = line_num
                data_list.append(item)
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON on line {line_num}")
    return data_list

def save_results(results: List[Dict], output_path: str, is_temp: bool = False, batch_num: int = 0):
    """将结果列表保存为JSONL文件"""
    results.sort(key=lambda r: r['line_number'])

    if is_temp:
        path, ext = os.path.splitext(output_path)
        output_dir = os.path.dirname(output_path)
        save_path = os.path.join(output_dir, f"temp_results_batch_{batch_num}.jsonl")
    else:
        save_path = output_path
        
    output_dir = os.path.dirname(save_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(save_path, 'w', encoding='utf-8') as f:
        for result in results:
             f.write(json.dumps(result, ensure_ascii=False) + '\n')
             
    if is_temp:
        print(f"\nIntermediate results for {len(results)} items saved to {save_path}\n")
    else:
        print(f"\nFinal results for {len(results)} items saved to {save_path}\n")


def main():
    args = parse_arguments()
    print(f"Starting script with arguments: {vars(args)}")

    try:
        llm_client = DashScopeLLMClient(
            api_key=args.api_key,
            model_name=args.model_name,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            timeout=args.timeout
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error initializing client: {e}")
        return

    cache_manager = CacheManager(args.cache_dir, args.use_cache)
    data_list = load_dataset(args.dataset_path)
    data_list = data_list
    if not data_list:
        print("No data to process. Exiting.")
        return
    
    print(f"Loaded {len(data_list)} items to process.")

    start_time = time.time()
    all_results = []
    
    if args.use_async:
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        all_results = asyncio.run(run_async_processing(data_list, llm_client, cache_manager, args))
    else:
        all_results = run_sync_processing_with_threading(data_list, llm_client, cache_manager, args)

    if all_results:
        save_results(all_results, args.output_path)
        
        cached_count = sum(1 for r in all_results if r.get('from_cache', False))
        total_processed = len(all_results)
        error_count = sum(1 for r in all_results if isinstance(r.get('gen_ans'), str) and r['gen_ans'].startswith("Error:"))
        
        print("\n--- Processing Summary ---")
        print(f"Total items processed: {total_processed}")
        if args.use_cache:
            print(f"Cache hits: {cached_count}/{total_processed} ({cached_count / total_processed * 100:.1f}%)")
        print(f"Errors encountered: {error_count}")

    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()