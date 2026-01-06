"""
HY-MT Translation Node for ComfyUI
Author: Your Name
Description: A node for translating text using HY-MT translation models from Tencent
"""

import os
import sys
import requests
from tqdm import tqdm
import zipfile
import tarfile

# 临时模拟torchao模块，避免与当前PyTorch ROCm版本冲突
class MockModule:
    def __init__(self, name):
        self.__name__ = name
    
    def __getattr__(self, name):
        return MockModule(f"{self.__name__}.{name}")

sys.modules['torchao'] = MockModule('torchao')
sys.modules['torchao.float8'] = MockModule('torchao.float8')
sys.modules['torchao.float8.inference'] = MockModule('torchao.float8.inference')
sys.modules['torchao.float8.distributed_utils'] = MockModule('torchao.float8.distributed_utils')
sys.modules['torchao.dtypes'] = MockModule('torchao.dtypes')
sys.modules['torchao.dtypes.floatx'] = MockModule('torchao.dtypes.floatx')
sys.modules['torchao.dtypes.affine_quantized_tensor_ops'] = MockModule('torchao.dtypes.affine_quantized_tensor_ops')
sys.modules['torchao.dtypes.floatx.cutlass_semi_sparse_layout'] = MockModule('torchao.dtypes.floatx.cutlass_semi_sparse_layout')
sys.modules['torchao.dtypes.floatx.float8_layout'] = MockModule('torchao.dtypes.floatx.float8_layout')
sys.modules['torchao.float8.float8_linear_utils'] = MockModule('torchao.float8.float8_linear_utils')
sys.modules['torchao.float8.float8_linear'] = MockModule('torchao.float8.float8_linear')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import folder_paths

# 尝试导入 llama-cpp-python
try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False
    print("Warning: llama-cpp-python not installed. GGUF model support will be disabled.")

# Add the custom nodes directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 模型下载配置
SAFETENSORS_MODELS = {
    "HY-MT1.5-1.8B": "https://huggingface.co/tencent/HY-MT1.5-1.8B/resolve/main/",
    "HY-MT1.5-1.8B-FP8": "https://huggingface.co/tencent/HY-MT1.5-1.8B-FP8/resolve/main/",
    "HY-MT1.5-7B": "https://huggingface.co/tencent/HY-MT1.5-7B/resolve/main/",
    "HY-MT1.5-7B-FP8": "https://huggingface.co/tencent/HY-MT1.5-7B-FP8/resolve/main/"
}

GGUF_MODELS = {
    "Hunyuan-MT-7B.Q4_K_M.gguf": "https://huggingface.co/mradermacher/Hunyuan-MT-7B-GGUF/resolve/main/Hunyuan-MT-7B.Q4_K_M.gguf?download=true"
}

def download_file(url, save_path):
    """
    Download a file from a URL with progress bar
    """
    try:
        print(f"Downloading: {url}")
        print(f"Save to: {save_path}")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 发送请求
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        # 获取文件大小
        total_size = int(response.headers.get('content-length', 0))
        
        # 下载文件
        with open(save_path, 'wb') as file:
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(save_path)) as pbar:
                for data in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                    size = file.write(data)
                    pbar.update(size)
        
        print(f"Download completed: {save_path}")
        return True
    except Exception as e:
        print(f"Download error: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return False

def download_safetensors_model(model_name, model_dir):
    """
    Download safetensors model from Hugging Face
    """
    model_path = os.path.join(model_dir, model_name)
    
    if os.path.exists(model_path) and os.listdir(model_path):
        print(f"Model {model_name} already exists, skipping download")
        return True
    
    if model_name not in SAFETENSORS_MODELS:
        print(f"Model {model_name} not in supported list")
        return False
    
    base_url = SAFETENSORS_MODELS[model_name]
    files_to_download = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json"
    ]
    
    os.makedirs(model_path, exist_ok=True)
    
    success_count = 0
    for filename in files_to_download:
        file_url = base_url + filename
        save_path = os.path.join(model_path, filename)
        
        if download_file(file_url, save_path):
            success_count += 1
    
    if success_count == len(files_to_download):
        print(f"Model {model_name} downloaded successfully")
        return True
    else:
        print(f"Failed to download all files for model {model_name}")
        # 清理不完整的下载
        try:
            import shutil
            shutil.rmtree(model_path)
        except:
            pass
        return False

def download_gguf_model(model_name, model_dir):
    """
    Download GGUF model from Hugging Face
    """
    model_path = os.path.join(model_dir, model_name)
    
    if os.path.exists(model_path):
        print(f"GGUF model {model_name} already exists, skipping download")
        return True
    
    if model_name not in GGUF_MODELS:
        print(f"GGUF model {model_name} not in supported list")
        return False
    
    model_url = GGUF_MODELS[model_name]
    
    return download_file(model_url, model_path)

class HYMTTranslator:
    """
    A class to handle HY-MT translation model loading and inference
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model = None
        self.models_dir = os.path.join(folder_paths.models_dir, "LLM", "HY-MT")
        
    def load_model(self, model_name):
        """
        Load the specified HY-MT translation model
        """
        import os  # Ensure os module is imported at the function level
        import json
        from transformers import AutoConfig
        
        if self.current_model == model_name and self.model is not None:
            return  # Model already loaded
        
        model_path = os.path.join(self.models_dir, model_name)
        
        if not os.path.exists(model_path) or not os.listdir(model_path):
            print(f"Model {model_name} not found locally, attempting to download...")
            success = download_safetensors_model(model_name, self.models_dir)
            if not success:
                raise FileNotFoundError(f"Failed to download model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model with quantization config handling
        try:
            # Try to load model with compressed-tensors quantization support
            print("Attempting to load model with quantization support...")
            
            # Check if model has quantization config
            config_file = os.path.join(model_path, "config.json")
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)
                
            has_quantization = "quantization_config" in config_data
            
            if has_quantization:
                print("Loading quantized model...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    device_map="auto",
                    trust_remote_code=True,
                    dtype=torch.bfloat16  # Match the model's configured dtype
                )
            else:
                # Load regular model
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    device_map="auto",
                    load_in_4bit=False,
                    load_in_8bit=False,
                    quantization_config=None
                )
                
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Attempting to load model without quantization config by modifying config file temporarily...")
            # Read the config file
            config_file = os.path.join(model_path, "config.json")
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)
                
            # Remove quantization config if present
            if "quantization_config" in config_data:
                del config_data["quantization_config"]
                print("Quantization config removed from temporary config")
                
            # Create a temporary config file
            temp_config_file = os.path.join(model_path, "temp_config.json")
            with open(temp_config_file, "w", encoding="utf-8") as f:
                json.dump(config_data, f)
                
            try:
                # Load config from temporary file
                config = AutoConfig.from_pretrained(temp_config_file, local_files_only=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    device_map="auto",
                    load_in_4bit=False,
                    load_in_8bit=False,
                    quantization_config=None,
                    config=config,
                    dtype=torch.bfloat16
                )
                print("Model loaded successfully with temporary config")
            except Exception as temp_e:
                print(f"Error loading model with temporary config: {temp_e}")
                # If all else fails, try to load without any config
                print("Attempting to load model without config...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    device_map="auto",
                    load_in_4bit=False,
                    load_in_8bit=False,
                    quantization_config=None,
                    ignore_mismatched_sizes=True
                )
                
            finally:
                # Clean up temporary config file
                if os.path.exists(temp_config_file):
                    os.remove(temp_config_file)
            
        self.current_model = model_name
        
    def generate_prompt(self, source_text, target_language, prompt_template=None):
        """
        Generate the appropriate prompt based on the source text and target language
        using simple and direct format that ensures concise and accurate translations
        """
        if prompt_template:
            # If custom prompt template is provided, use it with variable substitution
            return prompt_template.format(
                target_language=target_language,
                source_text=source_text
            )
        
        # 使用更简洁的提示格式，确保模型直接输出翻译结果
        default_template = "翻译:{source_text} -> {target_language}"
        
        return default_template.format(
            target_language=target_language,
            source_text=source_text
        )
        
    def translate(self, model_name, source_text, target_language, prompt_template=None, regex_pattern=None, timeout=30, max_new_tokens=512):
        """
        Translate text using the specified model and parameters with timeout
        """
        import time
        import threading
        from concurrent.futures import ThreadPoolExecutor, TimeoutError
        
        start_time = time.time()
        
        print(f"[Translation] Starting translation: model={model_name}, "
              f"source_text={repr(source_text[:100])}, target_language={target_language}")
        
        if not source_text.strip():
            print(f"[Translation] Empty source text, returning empty string (time: {time.time()-start_time:.2f}s)")
            return ""
            
        self.load_model(model_name)
        
        # Validate max_new_tokens parameter
        try:
            max_new_tokens = int(max_new_tokens)
            if max_new_tokens <= 0:
                raise ValueError("max_new_tokens must be a positive integer")
        except (ValueError, TypeError):
            print(f"[Translation] Invalid max_new_tokens value: {max_new_tokens}, using default value 512")
            max_new_tokens = 512
        
        # Generate prompt with template
        formatted_prompt = self.generate_prompt(source_text, target_language, prompt_template)
        print(f"[Translation] Generated prompt: {repr(formatted_prompt[:100])}...")
        
        # Tokenize the prompt directly with minimal options
        try:
            print(f"[Translation] Tokenizing prompt...")
            # Use simple tokenization without chat template to avoid errors
            tokenized_prompt = self.tokenizer(
                formatted_prompt,
                add_special_tokens=True,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=2048
            )
            
            print(f"[Translation] Tokenization completed, input_ids shape: {tokenized_prompt['input_ids'].shape}")
            
            # Generate translation
            # input_ids should always be integer type for embedding layer
            input_ids = tokenized_prompt['input_ids'].to(self.model.device, dtype=torch.long)
            attention_mask = tokenized_prompt['attention_mask'].to(self.model.device)
            
            print(f"[Translation] Inputs moved to device: {self.model.device}")
            
            # Match attention mask data type with model's dtype
            if hasattr(self.model, 'dtype'):
                attention_mask = attention_mask.to(self.model.dtype)
            elif hasattr(self.model.config, 'dtype') and self.model.config.dtype:
                target_dtype = getattr(torch, self.model.config.dtype)
                attention_mask = attention_mask.to(target_dtype)
            
            print(f"[Translation] Generating translation...")
            gen_start_time = time.time()
            
            # Define translation function for timeout
            def generate_translation():
                return self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    top_k=20,
                    top_p=0.6,
                    repetition_penalty=1.05,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    no_repeat_ngram_size=3,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Use a separate thread for translation to allow timeout
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(generate_translation)
            
            try:
                # Wait for translation with timeout
                outputs = future.result(timeout=timeout)
            except TimeoutError:
                print(f"[Translation] Translation timed out after {timeout} seconds")
                return ""
            
            gen_time = time.time() - gen_start_time
            print(f"[Translation] Generation completed in {gen_time:.2f} seconds")
            
            # Decode and extract the output
            print(f"[Translation] Decoding output...")
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"[Translation] Raw output: '{output_text}'")
            
            # 直接使用原始推理内容，不进行默认提取
            print(f"[Translation] Raw generated output: '{output_text}'")
            
            # 正则表达式提取功能
            if regex_pattern:
                import re
                # 处理多行正则表达式，每行一个表达式，按顺序匹配
                regex_patterns = [p.strip() for p in regex_pattern.split('\n') if p.strip()]
                
                for pattern in regex_patterns:
                    print(f"[Translation] Trying regex pattern '{pattern}'...")
                    # 尝试使用正则表达式提取内容
                    match = re.search(pattern, output_text)
                    if match:
                        # 如果有分组捕获，返回捕获的内容
                        if match.groups():
                            extracted_text = match.group(1).strip()
                            print(f"[Translation] Extracted text using regex '{pattern}': '{extracted_text}'")
                            # 如果提取到内容，返回提取的内容
                            if extracted_text:
                                return extracted_text
                        # 如果没有分组捕获，返回整个匹配内容
                        else:
                            extracted_text = match.group(0).strip()
                            print(f"[Translation] Extracted text using regex '{pattern}': '{extracted_text}'")
                            if extracted_text:
                                return extracted_text
                    else:
                        print(f"[Translation] No match found with regex '{pattern}'")
            
            # 如果没有使用正则表达式或提取失败，输出完整的推理内容
            print(f"[Translation] Outputting full generated text (no valid extraction)")
            return output_text
            
        except Exception as e:
            print(f"[Translation] Translation generation error: {e}")
            # Print detailed traceback for debugging
            import traceback
            print(f"[Translation] Detailed error traceback: {traceback.format_exc()}")
            return ""

class HYMTTranslatorGGUF:
    """
    A class to handle HY-MT translation model loading and inference using GGUF format
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model = None
        self.models_dir = os.path.join(folder_paths.models_dir, "LLM", "HY-MT")
        
    def load_model(self, model_name):
        """
        Load the specified HY-MT translation model in GGUF format
        """
        import os  # Ensure os module is imported at the function level
        
        if not HAS_LLAMA_CPP:
            raise ImportError("llama-cpp-python not installed. Please install it to use GGUF models.")
            
        if self.current_model == model_name and self.model is not None:
            return  # Model already loaded
        
        model_path = os.path.join(self.models_dir, model_name)
        
        if not os.path.exists(model_path):
            print(f"GGUF model {model_name} not found locally, attempting to download...")
            success = download_gguf_model(model_name, self.models_dir)
            if not success:
                raise FileNotFoundError(f"Failed to download GGUF model: {model_name}")
        
        print(f"Loading GGUF model from: {model_path}")
        
        # Load model using Llama from llama-cpp-python
        self.model = Llama(
            model_path=model_path,
            n_ctx=2048,  # Context window size
            n_batch=512,  # Batch size
            verbose=True
        )
        
        self.current_model = model_name
    
    def generate_prompt(self, source_text, target_language, prompt_template=None):
        """
        Generate the appropriate prompt based on the source text and target language
        """
        if prompt_template:
            # If custom prompt template is provided, use it with variable substitution
            return prompt_template.format(
                target_language=target_language,
                source_text=source_text
            )
        
        prompt_template = "翻译:{source_text} -> {target_language}"
        
        return prompt_template.format(
            target_language=target_language,
            source_text=source_text
        )
        
    def translate(self, model_name, source_text, target_language, prompt_template=None, regex_pattern=None, timeout=30, max_new_tokens=512):
        """
        Translate text using the specified model and parameters with timeout
        """
        import time
        import threading
        from concurrent.futures import ThreadPoolExecutor, TimeoutError
        
        start_time = time.time()
        
        print(f"[Translation] Starting translation: model={model_name}, "
              f"source_text={repr(source_text[:100])}, target_language={target_language}")
        
        if not source_text.strip():
            print(f"[Translation] Empty source text, returning empty string (time: {time.time()-start_time:.2f}s)")
            return ""
            
        self.load_model(model_name)
        
        # Validate max_new_tokens parameter
        try:
            max_new_tokens = int(max_new_tokens)
            if max_new_tokens <= 0:
                raise ValueError("max_new_tokens must be a positive integer")
        except (ValueError, TypeError):
            print(f"[Translation] Invalid max_new_tokens value: {max_new_tokens}, using default value 512")
            max_new_tokens = 512
        
        formatted_prompt = self.generate_prompt(source_text, target_language, prompt_template)
        print(f"[Translation] Generated prompt: {repr(formatted_prompt[:100])}...")
        
        try:
            print(f"[Translation] Generating translation...")
            gen_start_time = time.time()
            
            def generate_translation():
                return self.model(
                    formatted_prompt,
                    max_tokens=max_new_tokens,
                    temperature=0.7,
                    top_k=20,
                    top_p=0.6,
                    repeat_penalty=1.05,
                    stop=["</s>"],
                    echo=False
                )
            
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(generate_translation)
            
            try:
                outputs = future.result(timeout=timeout)
            except TimeoutError:
                print(f"[Translation] Translation timed out after {timeout} seconds")
                return ""
            
            gen_time = time.time() - gen_start_time
            print(f"[Translation] Generation completed in {gen_time:.2f} seconds")
            
            # Extract the translated text
            output_text = outputs["choices"][0]["text"].strip()
            print(f"[Translation] Raw output: '{output_text}'")
            
            if regex_pattern:
                import re
                regex_patterns = [p.strip() for p in regex_pattern.split('\n') if p.strip()]
                
                for pattern in regex_patterns:
                    print(f"[Translation] Trying regex pattern '{pattern}'...")
                    match = re.search(pattern, output_text)
                    if match:
                        if match.groups():
                            extracted_text = match.group(1).strip()
                            print(f"[Translation] Extracted text using regex '{pattern}': '{extracted_text}'")
                            if extracted_text:
                                return extracted_text
                        else:
                            extracted_text = match.group(0).strip()
                            print(f"[Translation] Extracted text using regex '{pattern}': '{extracted_text}'")
                            if extracted_text:
                                return extracted_text
                    else:
                        print(f"[Translation] No match found with regex '{pattern}'")
            
            print(f"[Translation] Outputting full generated text (no valid extraction)")
            return output_text
            
        except Exception as e:
            print(f"[Translation] Translation generation error: {e}")
            import traceback
            print(f"[Translation] Detailed error traceback: {traceback.format_exc()}")
            return ""

# Initialize the translators
translator = HYMTTranslator()
translator_gguf = HYMTTranslatorGGUF()

class HYMTTranslateNodeGGUF:
    """
    ComfyUI node for HY-MT translation using GGUF format models
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Define the input types for the node
        """
        models_dir = os.path.join(folder_paths.models_dir, "LLM", "HY-MT")
        available_models = []
        
        if os.path.exists(models_dir):
            available_models = [f for f in os.listdir(models_dir) if f.endswith(".gguf")]
            
        # Default GGUF model if none are found - only include models with auto-download support
        if not available_models:
            available_models = ["Hunyuan-MT-7B.Q4_K_M.gguf"]
        
        # Language options based on the documentation
        language_options = [
            ("zh", "中文"),
            ("en", "英语"),
            ("fr", "法语"),
            ("pt", "葡萄牙语"),
            ("es", "西班牙语"),
            ("ja", "日语"),
            ("tr", "土耳其语"),
            ("ru", "俄语"),
            ("ar", "阿拉伯语"),
            ("ko", "韩语"),
            ("th", "泰语"),
            ("it", "意大利语"),
            ("de", "德语"),
            ("vi", "越南语"),
            ("ms", "马来语"),
            ("id", "印尼语"),
            ("tl", "菲律宾语"),
            ("hi", "印地语"),
            ("zh-Hant", "繁体中文"),
            ("pl", "波兰语"),
            ("cs", "捷克语"),
            ("nl", "荷兰语"),
            ("km", "高棉语"),
            ("my", "缅甸语"),
            ("fa", "波斯语"),
            ("gu", "古吉拉特语"),
            ("ur", "乌尔都语"),
            ("te", "泰卢固语"),
            ("mr", "马拉地语"),
            ("he", "希伯来语"),
            ("bn", "孟加拉语"),
            ("ta", "泰米尔语"),
            ("uk", "乌克兰语"),
            ("bo", "藏语"),
            ("kk", "哈萨克语"),
            ("mn", "蒙古语"),
            ("ug", "维吾尔语"),
            ("yue", "粤语")
        ]
        
        # Format language options as "code - name" for better readability
        language_choices = [f"{lang[0]} - {lang[1]}" for lang in language_options]
        
        return {
            "required": {
                "gguf_name": (available_models,),
                "source_text": ("STRING", {"multiline": True, "default": "请输入要翻译的文本"}),
                "target_language": (
                    language_choices,
                    {"default": "en - 英语"}
                ),
                "prompt_template": ("STRING", {"multiline": True, "default": "把这词句翻译为{target_language}: {source_text} ", "tooltip": "自定义提示词模板，支持{source_text}和{target_language}变量"}),
                "regex_pattern": ("STRING", {"multiline": True, "default": "", "tooltip": "正则表达式模式，用于提取翻译结果"}),
                "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 2048, "tooltip": "生成新标记的最大数量，无效值将使用默认值512"}),
                "timeout": ("INT", {"default": 30, "min": 1, "max": 300, "tooltip": "翻译超时时间（秒）"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "prompt_text")
    FUNCTION = "translate"
    OUTPUT_NODE = True
    CATEGORY = "🦜 HY-MT"
    TITLE = "🦜 HY-MT-Translation (GGUF)"

    def translate(self, gguf_name, source_text, target_language, prompt_template=None, regex_pattern=None, max_new_tokens=512, timeout=30):
        """
        Translate text using HY-MT translation model in GGUF format
        """
        if not HAS_LLAMA_CPP:
            print("Error: llama-cpp-python not installed. Please install it to use GGUF models.")
            return ("", "")
            
        try:
            # 源文本预处理：合并多行，用逗号隔开，合并多个连续逗号
            processed_text = self.preprocess_text(source_text)
            
            # Parse language code from display string (format: "en - 英语")
            lang_code = target_language.split(" - ")[0]
            
            # Get language name from language code
            language_options = [
                ("zh", "中文"),
                ("en", "英语"),
                ("fr", "法语"),
                ("pt", "葡萄牙语"),
                ("es", "西班牙语"),
                ("ja", "日语"),
                ("tr", "土耳其语"),
                ("ru", "俄语"),
                ("ar", "阿拉伯语"),
                ("ko", "韩语"),
                ("th", "泰语"),
                ("it", "意大利语"),
                ("de", "德语"),
                ("vi", "越南语"),
                ("ms", "马来语"),
                ("id", "印尼语"),
                ("tl", "菲律宾语"),
                ("hi", "印地语"),
                ("zh-Hant", "繁体中文"),
                ("pl", "波兰语"),
                ("cs", "捷克语"),
                ("nl", "荷兰语"),
                ("km", "高棉语"),
                ("my", "缅甸语"),
                ("fa", "波斯语"),
                ("gu", "古吉拉特语"),
                ("ur", "乌尔都语"),
                ("te", "泰卢固语"),
                ("mr", "马拉地语"),
                ("he", "希伯来语"),
                ("bn", "孟加拉语"),
                ("ta", "泰米尔语"),
                ("uk", "乌克兰语"),
                ("bo", "藏语"),
                ("kk", "哈萨克语"),
                ("mn", "蒙古语"),
                ("ug", "维吾尔语"),
                ("yue", "粤语")
            ]
            
            lang_dict = dict(language_options)
            target_lang_name = lang_dict.get(lang_code, lang_code)
            
            # 生成格式化后的提示词
            formatted_prompt = translator_gguf.generate_prompt(processed_text, target_lang_name, prompt_template)
            
            translation = translator_gguf.translate(
                gguf_name,
                processed_text,
                target_lang_name,
                prompt_template,
                regex_pattern,
                timeout,
                max_new_tokens
            )
            
            return (translation, formatted_prompt)
            
        except Exception as e:
            print(f"Translation error: {e}")
            return ("", "")
            
    def preprocess_text(self, text):
        """
        Preprocess text by:
        1. Replacing newlines with commas
        2. Removing extra spaces around commas
        3. Merging multiple consecutive commas
        4. Trimming leading/trailing whitespace and commas
        """
        import re
        
        # 替换换行符为逗号
        processed = text.replace('\n', ',').replace('\r', ',')
        # 去除逗号周围的多余空格
        processed = re.sub(r'\s*,\s*', ',', processed)
        # 合并多个连续的逗号
        processed = re.sub(r',+', ',', processed)
        # 去除首尾的空格和逗号
        processed = processed.strip().strip(',')
        return processed

class HYMTTranslateNode:
    """
    ComfyUI node for HY-MT translation
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Define the input types for the node
        """
        models_dir = os.path.join(folder_paths.models_dir, "LLM", "HY-MT")
        available_models = []
        
        if os.path.exists(models_dir):
            available_models = [f for f in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, f))]
            
        # Default models if none are found (for reference) - only include models with auto-download support
        default_models = [
            "HY-MT1.5-1.8B",
            "HY-MT1.5-1.8B-FP8",
            "HY-MT1.5-7B",
            "HY-MT1.5-7B-FP8"
        ]
        
        # Combine available models with default models (remove duplicates)
        for m in default_models:
            if m not in available_models:
                available_models.append(m)
                
        # Language options based on the documentation
        language_options = [
            ("zh", "中文"),
            ("en", "英语"),
            ("fr", "法语"),
            ("pt", "葡萄牙语"),
            ("es", "西班牙语"),
            ("ja", "日语"),
            ("tr", "土耳其语"),
            ("ru", "俄语"),
            ("ar", "阿拉伯语"),
            ("ko", "韩语"),
            ("th", "泰语"),
            ("it", "意大利语"),
            ("de", "德语"),
            ("vi", "越南语"),
            ("ms", "马来语"),
            ("id", "印尼语"),
            ("tl", "菲律宾语"),
            ("hi", "印地语"),
            ("zh-Hant", "繁体中文"),
            ("pl", "波兰语"),
            ("cs", "捷克语"),
            ("nl", "荷兰语"),
            ("km", "高棉语"),
            ("my", "缅甸语"),
            ("fa", "波斯语"),
            ("gu", "古吉拉特语"),
            ("ur", "乌尔都语"),
            ("te", "泰卢固语"),
            ("mr", "马拉地语"),
            ("he", "希伯来语"),
            ("bn", "孟加拉语"),
            ("ta", "泰米尔语"),
            ("uk", "乌克兰语"),
            ("bo", "藏语"),
            ("kk", "哈萨克语"),
            ("mn", "蒙古语"),
            ("ug", "维吾尔语"),
            ("yue", "粤语")
        ]
        
        # Format language options as "code - name" for better readability
        language_choices = [f"{lang[0]} - {lang[1]}" for lang in language_options]
        
        # Create mapping from display name to language code
        lang_code_map = {f"{lang[0]} - {lang[1]}": lang[0] for lang in language_options}
        
        return {
            "required": {
                "ckpt_name": (available_models, {"default": "HY-MT1.5-1.8B-FP8"}),
                "source_text": ("STRING", {"default": "", "multiline": True}),
                "target_language": (
                    language_choices,
                    {"default": "en - 英语", "label_to_name": lang_code_map}
                ),
                "prompt_template": ("STRING", {"multiline": True, "default": "翻译:{source_text} -> {target_language}", "tooltip": "自定义提示词模板，支持{source_text}和{target_language}变量"}),
                "regex_pattern": ("STRING", {"default": "", "multiline": True}),
                "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 2048, "tooltip": "生成新标记的最大数量，无效值将使用默认值512"}),
                "timeout": ("INT", {"default": 30, "min": 1, "max": 300, "tooltip": "翻译超时时间（秒）"}),
            }
        }
        
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "prompt_text")
    FUNCTION = "translate"
    CATEGORY = "🦜 HY-MT"
    DESCRIPTION = "Translate text using HY-MT translation models. \n\nFeatures:\n- Multi-line regex extraction support: each line is a regex pattern, tested in order\n- Fallback to raw output if no regex patterns match\n- Supports extraction with capturing groups"
    
    def translate(self, ckpt_name, source_text, target_language, prompt_template=None, regex_pattern=None, max_new_tokens=512, timeout=30):
        """
        Perform translation
        """
        if not source_text.strip():
            return ("", "")
            
        try:
            # 源文本预处理：合并多行，用逗号隔开，合并多个连续逗号
            processed_text = self.preprocess_text(source_text)
            
            # Parse language code from display string (format: "en - 英语")
            lang_code = target_language.split(" - ")[0]
            
            # Get language name from language code
            language_options = [
                ("zh", "中文"),
                ("en", "英语"),
                ("fr", "法语"),
                ("pt", "葡萄牙语"),
                ("es", "西班牙语"),
                ("ja", "日语"),
                ("tr", "土耳其语"),
                ("ru", "俄语"),
                ("ar", "阿拉伯语"),
                ("ko", "韩语"),
                ("th", "泰语"),
                ("it", "意大利语"),
                ("de", "德语"),
                ("vi", "越南语"),
                ("ms", "马来语"),
                ("id", "印尼语"),
                ("tl", "菲律宾语"),
                ("hi", "印地语"),
                ("zh-Hant", "繁体中文"),
                ("pl", "波兰语"),
                ("cs", "捷克语"),
                ("nl", "荷兰语"),
                ("km", "高棉语"),
                ("my", "缅甸语"),
                ("fa", "波斯语"),
                ("gu", "古吉拉特语"),
                ("ur", "乌尔都语"),
                ("te", "泰卢固语"),
                ("mr", "马拉地语"),
                ("he", "希伯来语"),
                ("bn", "孟加拉语"),
                ("ta", "泰米尔语"),
                ("uk", "乌克兰语"),
                ("bo", "藏语"),
                ("kk", "哈萨克语"),
                ("mn", "蒙古语"),
                ("ug", "维吾尔语"),
                ("yue", "粤语")
            ]
            
            lang_dict = dict(language_options)
            target_lang_name = lang_dict.get(lang_code, lang_code)
            
            # 生成格式化后的提示词
            formatted_prompt = translator.generate_prompt(processed_text, target_lang_name, prompt_template)
            
            # Perform translation
            translation = translator.translate(ckpt_name, processed_text, target_lang_name, prompt_template, regex_pattern, timeout, max_new_tokens)
            return (translation, formatted_prompt)
            
        except Exception as e:
            print(f"Translation error: {e}")
            return ("", "")
            
    def preprocess_text(self, text):
        """
        Preprocess text by:
        1. Replacing newlines with commas
        2. Removing extra spaces around commas
        3. Merging multiple consecutive commas
        4. Trimming leading/trailing whitespace and commas
        """
        import re
        
        # 替换换行符为逗号
        processed = text.replace('\n', ',').replace('\r', ',')
        # 去除逗号周围的多余空格
        processed = re.sub(r'\s*,\s*', ',', processed)
        # 合并多个连续的逗号
        processed = re.sub(r',+', ',', processed)
        # 去除首尾的空格和逗号
        processed = processed.strip().strip(',')
        return processed

# Register the nodes
NODE_CLASS_MAPPINGS = {
    "HY-MT-Translator": HYMTTranslateNode,
    "HY-MT-Translator-GGUF": HYMTTranslateNodeGGUF
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HY-MT-Translator": "🦜HY-MT-Translation",
    "HY-MT-Translator-GGUF": "🦜HY-MT-Translation (GGUF)"
}
