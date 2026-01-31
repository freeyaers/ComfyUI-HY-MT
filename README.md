👉 **[简体中文](https://github.com/freeyaers/ComfyUI-HY-MT/blob/main/README_CN.md)**

# ComfyUI-HY-MT

HY-MT Translator Node for ComfyUI - A ComfyUI plugin for text translation using Tencent's HY-MT translation models.

## Features

- Supports multiple HY-MT translation model versions (1.8B, 7B, including FP8 quantized versions)
- Supports GGUF format quantized models (e.g., Hunyuan-MT-7B.Q4_K_M.gguf)
- Supports 37 languages for translation
- Provides a flexible prompt system with support for custom translation prompts
- Source text preprocessing: automatically merges multiple lines, separates with commas, and merges multiple consecutive commas
- Supports custom timeout settings to prevent translation tasks from becoming unresponsive
- Supports regular expression extraction for precise extraction of translation results
- Automatically detects source language and generates appropriate translation prompts
- Supports batch translation and custom context translation
- **Automatic model download**: When the selected model does not exist, it automatically downloads from Hugging Face and saves to the local directory

## Installation

### 1. Manual Installation

1. Download or clone this repository to the `custom_nodes` directory of ComfyUI
2. Ensure your ComfyUI environment has the required dependencies installed
3. Place the HY-MT model files in the `ComfyUI/models/LLM/HY-MT/` directory

### 2. Dependencies

Ensure your Python environment has the following dependencies installed:

```bash
pip install transformers torch modelscope requests tqdm
```

### 3. GGUF Model Support

If you need to use GGUF format models, you also need to install additional dependencies:

```bash
pip install llama-cpp-python
```

## Model Preparation

### Automatic Download Feature

When you first use a model that supports automatic download, the plugin will automatically download the model from Hugging Face and save it to the local directory. Models that support automatic download include:

**Standard models (safetensors):**
- HY-MT1.5-1.8B - https://huggingface.co/tencent/HY-MT1.5-1.8B
- HY-MT1.5-1.8B-FP8 - https://huggingface.co/tencent/HY-MT1.5-1.8B-FP8
- HY-MT1.5-7B - https://huggingface.co/tencent/HY-MT1.5-7B
- HY-MT1.5-7B-FP8 - https://huggingface.co/tencent/HY-MT1.5-7B-FP8

**GGUF models:**
- Hunyuan-MT-7B.Q4_K_M.gguf - https://huggingface.co/mradermacher/Hunyuan-MT-7B-GGUF

### Manual Model Placement

If you prefer to download models manually, you can place the model files in the following directory:

```
ComfyUI/models/LLM/HY-MT/
├── HY-MT1.5-1.8B/              # Standard model folder (contains config.json, model.safetensors, etc.)
├── HY-MT1.5-1.8B-FP8/          # FP8 quantized version model folder
├── HY-MT1.5-7B-FP8/            # 7B FP8 quantized version model folder
└── Hunyuan-MT-7B.Q4_K_M.gguf   # GGUF format model file
```

## Usage Instructions

![Image text](https://github.com/freeyaers/ComfyUI-HY-MT/blob/main/workflows/HY-MT-Translation%20(GGUF).png)

![Image text](https://github.com/freeyaers/ComfyUI-HY-MT/blob/main/workflows/HY-MT-Translation.png)

### HY-MT Translator Node (Standard Model)

1. In ComfyUI, find the `🦜HY-MT` category and select the `🦜HY-MT-Translation` node
2. Select the translation model (recommended to use HY-MT1.5-1.8B-FP8 by default)
3. Enter the text content to be translated
4. Select the target language
5. (Optional) Enter a custom prompt template (if needed)
6. (Optional) Enter a regular expression pattern for extracting translation results in a specific format
7. (Optional) Set the translation timeout (default 30 seconds)
8. Connect the output to other nodes or directly view the translation result

### HY-MT Translator (GGUF) Node (GGUF Model)

1. In ComfyUI, find the `🦜 HY-MT` category and select the `🦜 HY-MT-Translation (GGUF)` node
2. Select the GGUF format translation model (e.g., Hunyuan-MT-7B.Q4_K_M.gguf)
3. Enter the text content to be translated
4. Select the target language
5. (Optional) Enter a custom prompt template (if needed)
6. (Optional) Enter a regular expression pattern for extracting translation results in a specific format
7. (Optional) Set the translation timeout (default 30 seconds)
8. Connect the output to other nodes or directly view the translation result

## Prompt Template Syntax

### Default Prompt Template

The default prompt template is:
```
Translation: {source_text} -> {target_language}
```

### Custom Prompt Template

You can customize the prompt template through the `prompt_template` parameter, which supports the following variables:

- `{source_text}` - Preprocessed source text (multiple lines merged into one, separated by commas)
- `{target_language}` - Target language name in Chinese (e.g., "English", "Japanese", etc.)

### Prompt Examples

#### Simple Translation
```
Translation: {source_text} -> {target_language}
```

#### More Detailed Prompt
```
Please translate the following content into {target_language}, keeping the original meaning accurate:
{source_text}
```

#### Professional Field Translation
```
Please translate the following technical document content into {target_language}, using accurate terminology:
{source_text}
```

#### Natural Language Style Translation
```
Please translate the following content into natural and fluent {target_language}:
{source_text}
```

## Parameter Description

### Input Parameters

| Parameter Name | Type | Description | Node |
|---------|------|------|------|
| `ckpt_name` | Dropdown | Select the standard model to use | HY-MT Translator |
| `gguf_name` | Dropdown | Select the GGUF format model to use | HY-MT Translator (GGUF) |
| `source_text` | Text box | Source text to translate, supports multi-line input | Both nodes |
| `target_language` | Dropdown | Target language, formatted as "language code - language name" | Both nodes |
| `prompt_template` | Text box | Custom prompt template, supports {source_text} and {target_language} variables | Both nodes |
| `regex_pattern` | Text box | Regular expression pattern for extracting translation results (supports multi-line mode) | Both nodes |
| `max_tokens` | Integer | Maximum number of new tokens to generate, range 1-4096, default 512 | Both nodes |
| `timeout` | Integer | Translation timeout (seconds), range 1-300, default 30 seconds | Both nodes |

### Output Parameters

| Parameter Name | Type | Description | Node |
|---------|------|------|------|
| `text` | String | Translated text result | Both nodes |
| `prompt_text` | String | Processed complete prompt (including content after variable substitution) | Both nodes |

## Supported Languages

| Language | Abbreviation | Chinese Name |
|------|------|----------|
| Chinese | zh | 中文 |
| English | en | 英语 |
| French | fr | 法语 |
| Portuguese | pt | 葡萄牙语 |
| Spanish | es | 西班牙语 |
| Japanese | ja | 日语 |
| Turkish | tr | 土耳其语 |
| Russian | ru | 俄语 |
| Arabic | ar | 阿拉伯语 |
| Korean | ko | 韩语 |
| Thai | th | 泰语 |
| Italian | it | 意大利语 |
| German | de | 德语 |
| Vietnamese | vi | 越南语 |
| Malay | ms | 马来语 |
| Indonesian | id | 印尼语 |
| Filipino | tl | 菲律宾语 |
| Hindi | hi | 印地语 |
| Traditional Chinese | zh-Hant | 繁体中文 |
| Polish | pl | 波兰语 |
| Czech | cs | 捷克语 |
| Dutch | nl | 荷兰语 |
| Khmer | km | 高棉语 |
| Burmese | my | 缅甸语 |
| Persian | fa | 波斯语 |
| Gujarati | gu | 古吉拉特语 |
| Urdu | ur | 乌尔都语 |
| Telugu | te | 泰卢固语 |
| Marathi | mr | 马拉地语 |
| Hebrew | he | 希伯来语 |
| Bengali | bn | 孟加拉语 |
| Tamil | ta | 泰米尔语 |
| Ukrainian | uk | 乌克兰语 |
| Tibetan | bo | 藏语 |
| Kazakh | kk | 哈萨克语 |
| Mongolian | mn | 蒙古语 |
| Uyghur | ug | 维吾尔语 |
| Cantonese | yue | 粤语 |

## Notes

1. The first time you use it, if the model is not locally available, it will be automatically downloaded
2. Once loaded, the model will remain in memory until ComfyUI is closed
3. If the source text field is empty, it will directly return an empty value and skip inference
4. Custom prompts take priority over automatically generated prompts
5. GGUF models require the llama-cpp-python library to be installed
6. Source text will be automatically preprocessed: merge multiple lines, separate with commas, and merge multiple consecutive commas
7. If regular expression matching fails, the original translation result will be returned

## Performance Optimization Suggestions

1. Using FP8 quantized versions of models can significantly reduce memory usage
2. For long text translation, you can appropriately adjust the max_tokens parameter
3. If translation tasks frequently timeout, you can appropriately increase the value of the timeout parameter

## Project Structure

```
ComfyUI-HY-MT/
├── __init__.py                         # Core implementation file
├── README.md                           # Project documentation
├── requirements.txt                    # Project dependencies file
├── comfyui_hy_mt_config.json           # Node configuration file
└── workflows/                          # Workflow examples
    ├── HY-MT-Translation.png           # Standard model workflow
    └── HY-MT-Translation (GGUF).png    # GGUF model workflow
```

## License

This project is distributed under the MIT License for personal and commercial use.

## Contact Information

If you have any questions or suggestions, please contact the developer through GitHub Issues.
