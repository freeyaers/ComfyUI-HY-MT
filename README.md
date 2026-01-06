# ComfyUI-HY-MT

HY-MT Translator Node for ComfyUI - 一个用于使用腾讯HY-MT翻译模型进行文本翻译的ComfyUI插件。

## 功能特点

- 支持多个HY-MT翻译模型版本（1.8B、7B，包含FP8量化版本）
- 支持GGUF格式的量化模型（如Hunyuan-MT-7B.Q4_K_M.gguf）
- 支持37种语言翻译
- 提供灵活的提示词系统，支持自定义翻译提示
- 源文本预处理功能：自动合并多行文本，用逗号隔开，并合并多个连续逗号
- 支持自定义超时设置，防止翻译任务长时间无响应
- 支持正则表达式提取功能，用于精确提取翻译结果
- 自动检测源语言并生成合适的翻译提示
- 支持批量翻译和自定义上下文翻译
- **自动模型下载**：当选择的模型不存在时，自动从Hugging Face下载并保存到本地目录

## 安装方法

### 1. 手动安装

1. 下载或克隆此仓库到ComfyUI的`custom_nodes`目录
2. 确保您的ComfyUI环境已安装所需的依赖库
3. 将HY-MT模型文件放置在`ComfyUI/models/LLM/HY-MT/`目录下

### 2. 依赖库

确保您的Python环境已安装以下依赖：

```bash
pip install transformers torch modelscope requests tqdm
```

### 3. GGUF模型支持

如果您需要使用GGUF格式的模型，还需要安装额外的依赖：

```bash
pip install llama-cpp-python
```

## 模型准备

### 自动下载功能

当您首次使用支持自动下载的模型时，插件会自动从Hugging Face下载模型并保存到本地目录。支持自动下载的模型包括：

**标准模型（safetensors）：**
- HY-MT1.5-1.8B - https://hf-mirror.com/Tencent-Hunyuan/Hunyuan-MT1.5-1.8B
- HY-MT1.5-1.8B-FP8 - https://hf-mirror.com/Tencent-Hunyuan/Hunyuan-MT1.5-1.8B-FP8
- HY-MT1.5-7B-FP8 - https://hf-mirror.com/Tencent-Hunyuan/Hunyuan-MT1.5-7B-FP8

**GGUF模型：**
- Hunyuan-MT-7B.Q4_K_M.gguf - https://hf-mirror.com/mradermacher/Hunyuan-MT-7B-GGUF

### 手动放置模型

如果您希望手动下载模型，可以将模型文件放置在以下目录：

```
ComfyUI/models/LLM/HY-MT/
├── HY-MT1.5-1.8B/              # 标准模型文件夹（包含config.json、model.safetensors等文件）
├── HY-MT1.5-1.8B-FP8/          # FP8量化版本模型文件夹
├── HY-MT1.5-7B-FP8/            # 7B FP8量化版本模型文件夹
└── Hunyuan-MT-7B.Q4_K_M.gguf   # GGUF格式模型文件
```



## 使用说明

![Image text](https://github.com/freeyaers/ComfyUI-HY-MT/blob/main/workflows/flowchart/HY-MT-Translation%20(GGUF).png)

![Image text](https://github.com/freeyaers/ComfyUI-HY-MT/blob/main/workflows/flowchart/HY-MT-Translation.png)

### HY-MT Translator 节点（标准模型）

1. 在ComfyUI中，找到`🦜HY-MT`分类下的`🦜HY-MT-Translation`节点
2. 选择翻译模型（默认推荐使用HY-MT1.5-1.8B-FP8）
3. 输入要翻译的文本内容
4. 选择目标语言
5. （可选）输入自定义提示词模板（如果需要）
6. （可选）输入正则表达式模式，用于提取特定格式的翻译结果
7. （可选）设置翻译超时时间（默认30秒）
8. 连接输出到其他节点或直接查看翻译结果

### HY-MT Translator (GGUF) 节点（GGUF模型）

1. 在ComfyUI中，找到`🦜 HY-MT`分类下的`🦜 HY-MT-Translation (GGUF)`节点
2. 选择GGUF格式的翻译模型（如Hunyuan-MT-7B.Q4_K_M.gguf）
3. 输入要翻译的文本内容
4. 选择目标语言
5. （可选）输入自定义提示词模板（如果需要）
6. （可选）输入正则表达式模式，用于提取特定格式的翻译结果
7. （可选）设置翻译超时时间（默认30秒）
8. 连接输出到其他节点或直接查看翻译结果

## 提示词模板语法

### 默认提示词模板

默认提示词模板为：
```
翻译：{source_text} -> {target_language}
```

### 自定义提示词模板

您可以通过`prompt_template`参数自定义提示词模板，支持以下变量：

- `{source_text}` - 预处理后的源文本（多行合并为一行，用逗号隔开）
- `{target_language}` - 目标语言的中文名称（如"英语"、"日语"等）

### 提示词示例

#### 简单翻译
```
翻译：{source_text} -> {target_language}
```

#### 更详细的提示
```
请将以下内容翻译成{target_language}，保持原意准确：
{source_text}
```

#### 专业领域翻译
```
请将以下技术文档内容翻译成{target_language}，术语使用要准确：
{source_text}
```

#### 自然语言风格翻译
```
请将以下内容翻译成自然流畅的{target_language}：
{source_text}
```

## 参数说明

### 输入参数

| 参数名称 | 类型 | 描述 | 节点 |
|---------|------|------|------|
| `ckpt_name` | 下拉菜单 | 选择要使用的标准模型 | HY-MT Translator |
| `gguf_name` | 下拉菜单 | 选择要使用的GGUF格式模型 | HY-MT Translator (GGUF) |
| `source_text` | 文本框 | 要翻译的源文本，支持多行输入 | 两个节点 |
| `target_language` | 下拉菜单 | 目标语言，格式为"语言代码 - 语言名称" | 两个节点 |
| `prompt_template` | 文本框 | 自定义提示词模板，支持{source_text}和{target_language}变量 | 两个节点 |
| `regex_pattern` | 文本框 | 正则表达式模式，用于提取翻译结果（支持多行模式） | 两个节点 |
| `timeout` | 整数 | 翻译超时时间（秒），范围1-300，默认30秒 | 两个节点 |

### 输出参数

| 参数名称 | 类型 | 描述 | 节点 |
|---------|------|------|------|
| `text` | 字符串 | 翻译后的文本结果 | 两个节点 |
| `prompt_text` | 字符串 | 处理后的完整提示词（包含变量替换后的内容） | 两个节点 |

## 支持的语言

| 语言 | 缩写 | 中文名称 |
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

## 注意事项

1. 第一次使用模型时，会自动加载到内存中，可能需要一些时间
2. 模型加载后会保持在内存中，直到ComfyUI关闭
3. 如果源文本字段为空，将直接返回空值，跳过推理
4. 自定义提示词优先级高于自动生成的提示词
5. FP8模型需要额外的配置步骤，请参考上面的说明
6. GGUF模型需要安装llama-cpp-python库支持
7. 源文本会自动进行预处理：合并多行，用逗号隔开，并合并多个连续逗号
8. 如果正则表达式匹配失败，会返回原始的翻译结果

## 性能优化建议

1. 使用FP8量化版本的模型可以显著减少内存占用
2. 对于长文本翻译，可以适当调整max_new_tokens参数
3. 可以根据需要调整生成参数（top_k、top_p、temperature等）
4. 对于GGUF模型，可以通过调整上下文窗口大小(n_ctx)和批处理大小(n_batch)来优化性能
5. 如果翻译任务经常超时，可以适当增加timeout参数的值

## 项目结构

```
ComfyUI-HY-MT/
├── __init__.py                         # 核心实现文件
├── README.md                           # 项目说明文档
├── requirements.txt                    # 项目依赖文件
├── comfyui_hy_mt_config.json           # 节点配置文件
└── workflows/                          # 工作流程示例
    ├── HY-MT-Translation.json          # 标准模型工作流程
    ├── HY-MT-Translation (GGUF).json   # GGUF模型工作流程
    └── flowchart/                      # 参考流程图
        ├── HY-MT-Translation.png
        └── HY-MT-Translation (GGUF).png
```

## 许可证

本项目根据MIT许可证分发，供个人和商业使用。

## 联系信息

如有问题或建议，请通过GitHub Issues联系开发者。
