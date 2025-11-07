import os
import json
import pandas as pd
from pydantic import BaseModel, Field
from academiclm import DocumentLM, MeasurementLM
from academiclm import get_filenames_in_directory

filepaths = get_filenames_in_directory('data/papers/')
pdf_filepaths = [os.path.join('data/papers/', f) for f in filepaths if f.endswith('.pdf')]

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
df = pd.DataFrame(data)
df.to_csv('data/experiments/forest_measurements.csv', index=False)