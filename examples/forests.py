import os
import json
import numpy as np
import pandas as pd
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from academiclm import MeasurementLM


# Convert PDF to text chunks
pdf_filepath = 'data/monitoring_tropical_forests.pdf'

pipeline_options = PdfPipelineOptions(do_table_structure=True)
pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
result = doc_converter.convert(pdf_filepath)
doc = result.document.export_to_markdown()

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=1024,
    chunk_overlap=0,
    separators=["\n\n", "\n", " "]
)
doc_chunks = text_splitter.split_text(doc)
text_chunks = [[t.strip() for t in doc_chunks if len(t.strip()) > 0]]


# Describe the specific entities to search for and identify.
measurement_types = {
    "Total carbon sequestered per annum": "Total carbon sequestered per annum (in units Mt C a^-1)",
}
identification_prompt = (
    "You are an expert in identifying countries with tropical forests are referenced in text from scientific literature. "
    "Using the given context, find and identify all countries it mentions, by name. "
   f"In particular, look for countries which are mentioned with respect to the following measurements: {list(measurement_types.keys())}. "
    "Format your response as a list of country names, separated by semicolons. "
    "If no countries are mentioned, respond only with 'None'. "
    "Do not include any additional text or explanation in your response."
)

# Extract measurements
measurementlm = MeasurementLM(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    identification_prompt=identification_prompt,
    measurement_types=measurement_types,
    test_entities=["Chile"], # Fake entity for testing
    sampling_params={
        "max_new_tokens" : 100,
    }
)

data = measurementlm.fit(text_chunks)

# Save and remove context and parametric scores:
filtered_data = []
for entry in data:
    context_scores = entry.pop('context_scores', None)
    parametric_scores = entry.pop('parametric_scores', None)
    filtered_data.append(entry)

    # Save as npz 
    measurement_id = entry['measurement_id']
    scores_dir = 'data/scores'

    np.savez_compressed(f"{scores_dir}/context_{measurement_id}.npz", 
                        context_scores=context_scores)
    np.savez_compressed(f"{scores_dir}/parametric_{measurement_id}.npz", 
                        parametric_scores=parametric_scores)

df = pd.DataFrame(filtered_data)
df.to_csv('data/forest_measurements.csv', index=False)