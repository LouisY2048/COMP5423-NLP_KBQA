# KBQA 知识库问答系统

基于NQ10K数据集的知识库问答(Knowledge Base Question Answering, KBQA)系统。该系统使用多种检索方法从知识库中检索相关文档，并使用大型语言模型生成答案。

## 项目结构

```
/
|---data/              # 存放数据集
|---backend/           # 存放后端代码
|   |---utils/         # 工具函数
|   |---retrieval/     # 文档检索模块
|   |---generation/    # 答案生成模块
|   |---main.py        # 主应用
|   |---evaluation.py  # 评估脚本
|---frontend/          # 存放前端代码
|   |---src/           # 前端源代码
|   |---public/        # 公共资源
|---metrics_calculation.py # 指标计算脚本
```

## 系统功能

该系统实现了三个主要功能：

1. **文档检索**
   - 关键词检索：使用BM25算法进行基于关键词的检索
   - 向量检索：使用预训练词嵌入模型进行基于向量的检索
   - 密集检索(DPR)：使用预训练语言模型进行基于密集向量的检索

2. **答案生成**
   - 使用Qwen2.5-7B-Instruct大型语言模型生成基于检索文档的答案

3. **用户界面**
   - 提供直观的用户界面，用户可以输入问题并选择检索方法
   - 显示检索到的相关文档和生成的答案

## 安装与运行

### 后端

1. 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

2. 运行服务器

```bash
cd backend
python main.py
```

### 前端

1. 安装依赖

```bash
cd frontend
npm install
```

2. 开发模式运行

```bash
cd frontend
npm run serve
```

3. 构建生产版本

```bash
cd frontend
npm run build
```

## 评估系统

使用提供的脚本评估系统性能：

```bash
cd backend
python evaluation.py --retrieval hybrid --split val
```

参数说明：
- `--retrieval`: 检索方法，可选值为 `keyword`, `vector`, `dense`, `hybrid`
- `--split`: 数据集分割，可选值为 `val`, `test`
- `--output`: 输出文件路径（可选）

## 系统架构

### 1. 检索模块

系统实现了三种文档检索方法：

- **关键词检索(BM25)**：传统的基于关键词匹配的检索方法
- **向量空间检索**：使用Sentence-BERT将文档和问题映射到向量空间，计算相似度
- **密集通道检索(DPR)**：使用预训练BERT模型单独对问题和文档进行编码，并使用FAISS进行高效的最近邻搜索

### 2. 答案生成模块

使用Qwen2.5-7B-Instruct模型，基于检索到的文档内容生成答案。通过精心设计的提示模板，引导模型生成准确、信息丰富的答案。

### 3. 前端界面

使用Vue.js构建的用户友好界面，支持问题输入、检索方法选择、展示检索文档和生成的答案。