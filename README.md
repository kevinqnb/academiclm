# AcademicLM :microscope: :books:

**Parse and analyze scientific research papers with large language models using mechanistic interpretability.**

PaperLM is a Python library that combines document processing with advanced language model analysis to extract insights from scientific papers. Built on top of [NNsight](https://github.com/ndif-ai/nnsight), it provides tools for detecting hallucinations in retrieval-augmented generation (RAG) systems and analyzing how language models process contextual information.

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/paperlm.git
cd paperlm

# Install with pixi (recommended)
pixi install

# Or install with pip
pip install -e .
```

### Basic Usage

```python
from paperlm import ContextLM

# Initialize the model
model = ContextLM(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    top_k=0.1,  # Top 10% of context tokens to analyze
    max_new_tokens=50
)

# Generate text with context analysis
context = "The Earth orbits around the Sun in an elliptical path."
instructions = "Explain planetary motion."

result = model.generate(context, instructions)

print(f"Response: {result['response']}")
print(f"Parametric Score: {result['parametric_score']:.4f}")
print(f"Context Score: {result['context_score']:.4f}")
```

## Core Concepts

PaperLM implements external context and parametric knowledge score methods from:
> Sun, Zhongxiang, et al. "ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation via Mechanistic Interpretability." ICLR. 2025.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
