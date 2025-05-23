## Homework Assignments

### HW1: Retrieval Augmented Generation with Agents

**Topic**: Building a question-answering system using Retrieval Augmented Generation (RAG) with LLM agents.

**Key Components**:
- **LLM Model**: Meta-Llama-3.1-8B-Instruct (Quantized Q8_0 version)
- **Search Tool**: Google Search integration with web scraping
- **Agent System**: Multiple specialized agents for different tasks
  - Question Extraction Agent
  - Keyword Extraction Agent  
  - QA Agent
- **RAG Pipeline**: Combines search results with LLM inference

**Technical Features**:
- GPU-accelerated inference using `llama-cpp-python`
- Asynchronous web scraping with `requests-html`
- Multi-agent architecture for improved accuracy
- Traditional Chinese language support
- Context window management (16,384 tokens)

**Dataset**:
- 30 public questions for development
- 60 private questions for evaluation
- Questions cover diverse topics including history, technology, culture, and current events

**Implementation Approaches**:
1. **Naive Approach**: Direct LLM inference without external knowledge
2. **Naive RAG**: Simple search + LLM combination
3. **RAG with Agents**: Multi-agent pipeline with specialized roles


## Setup and Usage

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for LLaMA inference)
- Google Colab (recommended environment)

### Installation
```bash
# Install required packages
pip install llama-cpp-python==0.3.4 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122
pip install googlesearch-python bs4 charset-normalizer requests-html lxml_html_clean
```

### Running HW1
1. Open `HW1/mlhw1.ipynb` in Google Colab
2. Mount Google Drive and set working directory
3. Run all cells to download model weights and datasets
4. Execute the RAG pipeline to generate answers

## File Descriptions

### HW1 Files
- `mlhw1.ipynb`: Complete implementation with RAG pipeline
- `Meta-Llama-3.1-8B-Instruct-Q8_0.gguf`: Quantized LLaMA model weights
- `public.txt`: 30 public questions with ground truth answers
- `private.txt`: 60 private evaluation questions
- `simonchu_*.txt`: Individual answer files for each question
- `simonchu.txt`: Final submission file with all 90 answers

## Performance Notes

- **Model Size**: ~8GB for quantized LLaMA 3.1 8B
- **Memory Requirements**: 16GB+ GPU VRAM recommended
- **Context Window**: 16,384 tokens maximum
- **Search Limitations**: Google Search rate limiting may occur
- **Language**: Optimized for Traditional Chinese responses