from pydantic import BaseModel
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from .contextlm import ContextLM


def response_validator(response_structure, response):
    pyd = response_structure.model_validate_json(response)
    out_dict = pyd.model_dump()
    return out_dict

class BooleanResponse(BaseModel):
    answer: bool



class MeasurementLM:
    """
    A language model class designed for organized collection of measurements from scientific text.

    Args:
        model_name (str): The name or path of the pre-trained language model from the huggingface 
            collection.
        item_description (str): Main description for the items to be measured.
        identification_schema (dict[str, str]): A dictionary defining the identification schema, 
            where keys are the measurement identifiers and values are their descriptions.
        measurement_schema (dict[str, str]): A dictionary defining the measurement schema, 
            where keys are

    Attributes:

    """
    def __init__(
        self,
        model_name: str,
        item_description: str,
        identification_schema: dict[str, str],
        measurement_schema: dict[str, str],
        sampling_params: dict[str, any] = None,
    ):
        self.model_name = model_name
        self.item_description = item_description
        self.identification_schema = identification_schema
        self.measurement_schema = measurement_schema
        self.sampling_params = {
            "temperature" : 0.90,
            "top_p" : 0.95,
            "top_k" : 64,
            "repetition_penalty" : 1.0,
            "max_tokens" : 2048,
        } | sampling_params

        self.llm = LLM(model=model_name)

    
    def _filter(self):
        """
        Filters the input text chunks to retain only those relevant to the item of interest.

        Args:
            
        Returns:
            
        """        
        messages = []
        for i, datapoint in enumerate(self.data):
            context = datapoint['context']
            prompt = (
                f"## Instructions:\n"
                f"Determine if the following context contains any information relevant to measuring "
                f"{', '.join(self.item_description)}. \n"
                f"Respond using a JSON object with a single key 'answer' and a boolean value "
                f"indicating relevance (true for relevant, false for irrelevant). "
                f"Example: {{\"answer\": false}}\n\n"
                f"## Context:\n"
                f"{context}"
            )
            messages.append([
                {"role": "user", "content": prompt}]
            )

        guided_decoding_params = GuidedDecodingParams(json=BooleanResponse.model_json_schema())
        sampling_params = SamplingParams(
            **self.sampling_params,
            guided_decoding=guided_decoding_params
        )

        responses = self.llm.chat(messages = messages, sampling_params = sampling_params)
        response_texts = [r.outputs[0].text for r in responses]
        response_validated = [
            response_validator(BooleanResponse, r) for r in response_texts
        ]

        filtered_data = []
        for i, resp in enumerate(response_validated):
            if resp['answer'] == True:
                filtered_data.append(self.data[i])
        
        return filtered_data
    

    def _identify(self):
        """
        Identifies items in the text chunks based on the identification schema.

        Args:
            
        Returns:
            
        """
        identification_schema_json = self.identification_schema.model_json_schema()
        identification_prompt = self.identification_schema.model_config['prompt']
        messages = []
        for i, datapoint in enumerate(self.data):
            context = datapoint['context']
            prompt = (
                f"## Instructions:\n"
                f"{identification_prompt}\n\n"
                f"## Context:\n"
                f"{context}"
            )
            messages.append([
                {"role": "user", "content": prompt}]
            )

        guided_decoding_params = GuidedDecodingParams(
            json=identification_schema_json
        )
        sampling_params = SamplingParams(
            **self.sampling_params,
            guided_decoding=guided_decoding_params
        )

        responses = self.llm.chat(messages = messages, sampling_params = sampling_params)
        response_texts = [r.outputs[0].text for r in responses]
        response_validated = []
        for r in response_texts:
            try:
                resp_validated = response_validator(self.identification_schema, r)
            except:
                resp_validated = {'items': []}
            response_validated.append(resp_validated)

        itemized_data = []
        for i, resp in enumerate(response_validated):
            for item in resp['items']:
                itemized_data.append(self.data[i] | item)

        return itemized_data
    

    def _measurements_filter(self):
        """
        Filters the input items to retain only those relevant for measurements.

        Args:
            
        Returns:
            
        """
        messages = []
        message_measurement_types = []
        message_data_ids = []
        for m in self.measurement_schema.model_fields.keys():
            m_description = self.measurement_schema.model_fields[m].description
            for i, datapoint in enumerate(self.data):
                context = datapoint['context']
                item = {k: v for k,v in datapoint.items() if k not in ['context', 'chunk_id']}
                prompt = (
                    f"## Instructions:\n"
                    f"Does the context give a value of {m_description} for the entity {item}? "
                    f"Do not respond positively unless there is explicit evidence in the context. "
                    f"Do not respond positively if the context reports a range of values, or an unspecific reference to a value. "
                    f"Respond using a JSON object with a single key 'answer' and a boolean value "
                    f"indicating relevance (true for relevant, false for irrelevant). "
                    f"Example: {{\"answer\": true}}\n\n"
                    f"## Context:\n"
                    f"{context}"
                )
                messages.append([
                    {"role": "user", "content": prompt}]
                )
                message_measurement_types.append(m)
                message_data_ids.append(i)

        guided_decoding_params = GuidedDecodingParams(json=BooleanResponse.model_json_schema())
        sampling_params = SamplingParams(
            **self.sampling_params,
            guided_decoding=guided_decoding_params
        )

        responses = self.llm.chat(messages = messages, sampling_params = sampling_params)
        response_texts = [r.outputs[0].text for r in responses]
        response_validated = [
            response_validator(BooleanResponse, r) for r in response_texts
        ]

        measurement_data = []
        for i, resp in enumerate(response_validated):
            if resp['answer'] == True:
                measurement_data.append(
                    self.data[message_data_ids[i]] | {'measurement': message_measurement_types[i]}
                )

        return measurement_data
    

    def _measure(self):
        """
        Extracts measurements from the text chunks for the identified items.

        Args:

        Returns:
            
        """
        messages = []
        for i, datapoint in enumerate(self.data):
            context = datapoint['context']
            item = {k: v for k,v in datapoint.items() if k not in ['context', 'chunk_id', 'measurement']}
            measurement = datapoint['measurement']
            instructions = (
                f"Extract the value of {measurement} for the entity {item} in the given context. "
                f"Respond with the value exactly as it is seen in the context. "
                f"Give the value only, and doo not include any additional text in your response. "
                f"Respond 'None' if the information is not explicitly available in the given context."
            )
            context = (
                f"{context}"
            )
            messages.append((context, instructions))

        ctxlm_params = {k: v for k,v in self.sampling_params.items() if k != 'max_tokens'}
        ctxlm_params['max_new_tokens'] = 20
        ctxlm = ContextLM(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            top_k = 0.05,
            sampling_params=ctxlm_params,
            verbose = False
        )
        measurement_responses = ctxlm.predict(messages)

        measured_data = []
        for i,response_dict in enumerate(measurement_responses):
            if response_dict['response'].strip().lower() != 'none':
                measured_data.append(
                    self.data[i] | 
                    {
                        'value': response_dict['response'],
                        'context_score': response_dict['context_score'],
                        'parametric_score': response_dict['parametric_score']
                    }
                )

        return measured_data

    def _standardize(self):
        """
        Gives standardized units to the extracted measurements.

        Args:

        Returns:
            
        """        
        messages = []
        message_data_ids = []
        for i, datapoint in enumerate(self.data):
            context = datapoint['context']
            item = {k: v for k,v in datapoint.items() if k not in ['context', 'chunk_id', 'measurement', 'value']}
            measurement = datapoint['measurement']
            measurement_val = datapoint['value']

            measurement_description = self.measurement_schema.model_fields[measurement].description
            available_units = self.measurement_schema.model_fields[measurement].json_schema_extra.get('units', None)

            if available_units is not None:
                units_str = ', '.join(available_units)
                prompt = (
                    f"## Instructions:\n"
                    f"Determine the units of measurement which the context uses for the value {measurement_val}, "
                    f"reported for {measurement_description} for the entity {item}. "
                    f"Your answer should be the best fitting unit from among the following options: {units_str}. "
                    f"If none of the options fit, respond with the unit 'other'. "
                    f"Respond with the unit only, and do not include any additional text.\n\n"
                    f"## Context:\n"
                    f"{context}"
                )
                messages.append([
                    {"role": "user", "content": prompt}]
                )
                message_data_ids.append(i)

        #guided_decoding_params = GuidedDecodingParams()
        sampling_params = SamplingParams(
            **self.sampling_params
        )
        responses = self.llm.chat(messages = messages, sampling_params = sampling_params)
        response_units = [r.outputs[0].text for r in responses]
        
        standardized_data = [datapoint for datapoint in self.data]
        for i, resp in enumerate(response_units):
            standardized_data[message_data_ids[i]]['units'] = resp.strip()

        return standardized_data
        


    def fit(
        self,
        chunks : list[str],
    ):
        """
        Fits the MeasurementLM to the provided text chunks by filtering, identifying items, 
        and extracting measurements.

        Args:
            chunks (list[str]): A list of text chunks.
        Returns:
            measurements (list[dict]): A list of measurements extracted for identified items.
        """
        self.data = [{'chunk_id': i, 'context' : c} for i, c in enumerate(chunks)]
        self.data = self._filter()
        self.data = self._identify()
        self.data = self._measurements_filter()
        self.data = self._measure()
        self.data = self._standardize()

        return self.data
    

    def save(self, filepath: str):
        """
        Saves the measurement data to a csv.

        Args:
            filepath (str): The path to the file where the data will be saved.
        """
        df = pd.DataFrame(self.data)
        df.to_csv(filepath, index=False)



