# AcademicLM :microscope: :books:

**Parse and analyze scientific research papers with large language models.**

*NOTE:* This project is a work in progress, and only a portion of it is shared here for the purpose of communicating my work. I kindly ask that you please be respectful of the content, for now it is shared for viewing and discussion only.

This library implements a system for extracting insights from scientific papers (which are in the form of pdfs) using large language models.
Specifically, we apply local and open source LLMs towards organized tasks for:
* Document OCR: translating pdf images into markdown, and splitting into paragraph sized chunks.
* Document extraction: systematically collecting data points from chunks of markdown text. 
* Hallucination detection

Our focus is on using small, local models for OCR and text generation tasks, and this library is designed to be compatible with any such model of your choosing. 

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/academiclm.git
cd academiclm

# Install with uv (recommended for working with VLLM)
uv sync
```

### Basic Usage

```python
from academiclm import DocumentLM, MeasurementLM

# First describe the specific entities to search for and identify.
identification_prompt = (
    "You are an expert in identifying countries with tropical forests are referenced in text from scientific literature. "
    "Using the given context, find, identify, and list all relevant countries it mentions."
)

class Identifier(BaseModel):
    name: str | None

class IdentificationSchema(BaseModel):
    items: list[Identifier]
    model_config = {
        'title': 'Identification Model',
        'prompt': identification_prompt
    }

# Next, outline a class of measurements to search for over each item identified
class MeasurementSchema(BaseModel):
    carbon_sequestration: float | None = Field(
        description="Total carbon sequestered per annum",
        json_schema_extra={'units': ["Mt C a^-1"]}
    )

# Read pdfs and separate text into paragraph sized chunks
doclm = DocumentLM(
    model = "allenai/olmOCR-2-7B-1025-FP8",
    ocr = True,
    ocr_prompt = "Transform the given pdf into markdown format.",
    sampling_params = {"temperature": 0.1, "max_tokens": 8192},
    chunk_size = 256, # Tokens per chunk
)
text_chunks = doclm.fit(pdf_filepaths)

# Extract measurements
measurementlm = MeasurementLM(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    item_description="Tropical Forests",
    identification_schema=IdentificationSchema,
    measurement_schema=MeasurementSchema,
    sampling_params={
        "temperature": 0.1,
        "top_p" : 0.95,
        "top_k" : 64,
        "max_tokens" : 4096,
        "seed": 342,
    }
)

data = measurementlm.fit(text_chunks)

```

## License

Copyright (c) 2025 [Kevin Quinn]. All rights reserved.

This repository and its contents are provided for viewing purposes only.
No part of this work may be reproduced, distributed, or used in any form
without the express written permission of the author.

