from pydantic import BaseModel
import pandas as pd
from .contextlm import ContextLM


def response_validator(response_structure, response):
    pyd = response_structure.model_validate_json(response)
    out_dict = pyd.model_dump()
    return out_dict

class BooleanResponse(BaseModel):
    answer: bool

class DataPointResponse(BaseModel):
    value: float | str | None
    units: str | None



class MeasurementLM:
    """
    A language model class designed for organized collection of measurements from scientific text.

    Args:
        model_name (str): The name or path of the pre-trained language model from the huggingface 
            collection.
        identification_schema (dict[str, str]): A dictionary defining the identification schema, 
            where keys are the measurement identifiers and values are their descriptions.
        measurement_schema (dict[str, str]): A dictionary defining the measurement schema, 
            where keys are
    """
    def __init__(
        self,
        model_name: str,
        identification_prompt: str,
        measurement_types : dict[str, str],
        test_entities: list[str] = None,
        sampling_params: dict[str, any] = None,
    ):
        self.model_name = model_name
        self.identification_prompt = identification_prompt
        self.measurement_types = measurement_types
        self.test_entities = test_entities
        self.sampling_params = {
            "max_new_tokens" : 100,
        } | sampling_params

        self.llm = ContextLM(
            model_name=model_name,
            sampling_params=self.sampling_params,
            generate_full_output=False,
            cache_dir="data/scores"
        )
    

    def _identify(self):
        """
        Identifies items in the text chunks based on the identification schema.

        Args:
            
        Returns:
            unique_itemized_data (list[dict]): A list of data points with identified items.
        """
        messages = []
        for i, datapoint in enumerate(self.data):
            instructions = self.identification_prompt
            context = datapoint['context']
            query = "Follow the instructions to identify the items mentioned in the context."
            messages.append((instructions, context, query))

        responses = self.llm.predict(messages)
        
        response_texts = [r['response'] for r in responses]
        response_validated = response_texts

        itemized_data = []
        for i, resp in enumerate(response_validated):
            datapoint = self.data[i]
            for item in resp.split(';'):
                item = item.strip()
                if item.lower() != 'none' and item != '':
                    itemized_data.append(
                        datapoint | {
                            'name': item
                        }
                    )

        # De-duplicate itemized data points
        unique_itemized_data = [dict(s) for s in {frozenset(d.items()) for d in itemized_data}]

        # Add in a fake forest name for testing scores:
        test_itemized_data = []
        for entry in unique_itemized_data:
            real_entry = entry
            test_itemized_data.append(real_entry)

            for test_entity in self.test_entities:
                test_entry = entry.copy()
                test_entry['name'] = test_entity
                test_itemized_data.append(test_entry)

        return test_itemized_data

    def _measure(self):
        """
        Extracts measurements from the text chunks for the identified items.

        Args:

        Returns:
            measured_data (list[dict]): A list of data points with extracted measurements.
        """
        messages = []
        message_measurement_types = []
        for measurement in self.measurement_types.keys():
            m_description = self.measurement_types[measurement]
            for i, datapoint in enumerate(self.data):
                item = {k: v for k,v in datapoint.items() if k not in ['context', 'paper_id', 'chunk_id']}
                instructions = (
                    f"You are an expert in extracting precise numerical data from user provided, scientific text. "
                    f"A value is a single numerical measurement explicitly mentioned in the context. "
                    f"You will be queried with a description of an specific entity to be measured, along with the measurement type to report for. "
                    f"Your task is to extract the corresponding value from the provided context. "
                    f"Copy the value exactly as it appears in the context. "
                    f"Respond 'None' if the requested information is not explicitly available in the given context. "
                    f"Do not include any additional text or explanation in your response."
                )
                context = datapoint['context']
                query = f"Extract the value of {m_description} for the entity {item}."
                messages.append((instructions, context, query))
                message_measurement_types.append(measurement)

        self.llm.set_output_mode(generate_full_output=True)
        responses = self.llm.predict(messages, ids = list(range(len(messages))))

        measured_data = []
        for i, measurement_dict in enumerate(responses):
            m_type = message_measurement_types[i]
            datapoint = self.data[i]
            response = measurement_dict['response']
            context_scores = measurement_dict['context_scores'] if 'context_scores' in measurement_dict else None
            parametric_scores = measurement_dict['parametric_scores'] if 'parametric_scores' in measurement_dict else None
            if response.strip().lower() != 'none':
                measured_data.append(
                    datapoint | {
                        'measurement': m_type,
                        'measurement_id': i,
                        'value': response.strip(),
                        'context_scores': context_scores,
                        'parametric_scores': parametric_scores,
                    }
                )
        return measured_data


    def fit(
        self,
        chunks : list[list[str]],
    ):
        """
        Fits the MeasurementLM to the provided text chunks by filtering, identifying items, 
        and extracting measurements.

        Args:
            chunks (list[list[str]]): A list of text chunks, for each paper.
        Returns:
            measurements (list[dict]): A list of measurements extracted for identified items.
        """
        self.data = []
        for i in range(len(chunks)):
            for j in range(len(chunks[i])):
                self.data.append({'paper_id': i, 'chunk_id': j, 'context' : chunks[i][j]})

        self.data = self._identify()
        self.data = self._measure()

        return self.data
    

    def save(self, filepath: str):
        """
        Saves the measurement data to a csv.

        Args:
            filepath (str): The path to the file where the data will be saved.
        """
        df = pd.DataFrame(self.data)
        df.to_csv(filepath, index=False)