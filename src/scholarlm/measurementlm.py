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

class DataPointResponse(BaseModel):
    value: float | str | None
    units: str | None



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
            instructions = (
                f"You are an expert at identifying relevant information in scientific texts. "
                f"Determine if the given context contains any relevant information. "
                f"Respond using a JSON object with a single key 'answer' and a boolean value "
                f"indicating relevance (true for relevant, false for irrelevant). "
                f"Example: {{\"answer\": false}}"
            )
            context = datapoint['context']
            query = "Is the context relevant to measuring or identifying " + f"{', '.join(self.item_description)}?"
            prompt = (
                f"## Instructions:\n{instructions}\n\n## Context:\n{context}\n\n## Query:\n{query}"
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
            instructions = identification_prompt
            context = datapoint['context']
            query = "Identify the items mentioned in the context."
            prompt = (
                f"## Instructions:\n{instructions}\n\n## Context:\n{context}\n\n## Query:\n{query}"
            )
            messages.append([
                {"role": "user","content": prompt}]
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
                item = {k: v for k,v in datapoint.items() if k not in ['context', 'chunk_id']}
                instructions = (
                    f"You are searching for data and are an expert in discerning whether or not a given piece of scientific text is relevant for your collection. "
                    f"You will be given a context from a research paper, along with a description of a feature to be evaluated for a specific entity. "
                    f"Your task is to determine if the context contains valued information for that feature and entity. "
                    f"Respond positive only if the context explicity provides data for the feature and entity in question. "
                    f"Respond using a JSON object with a single key 'answer' and a boolean value "
                    f"indicating relevance (true for relevant, false for irrelevant). "
                    f"Example: {{\"answer\": false}}"
                )
                context = datapoint['context']
                query = "Does the context contain data for " + f"{m_description}  the entity {item}?"
                prompt = (
                f"## Instructions:\n{instructions}\n\n## Context:\n{context}\n\n## Query:\n{query}"
                )
                messages.append([
                    {"role": "user","content": prompt}]
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
            item = {k: v for k,v in datapoint.items() if k not in ['context', 'chunk_id', 'measurement']}
            measurement = datapoint['measurement']
            instructions = (
                f"You are an expert in extracting precise data from scientific texts. "
                f"Given a requested feature type and an entity description, extract the corresponding data point from the provided context. "
                f"Copy the data point exactly as it appears in the context. "
                f"If the data point has a unit of measurement, include it with the value exactly as it is seen in the context. "
                f"Do not include any additional text in your response. "
                f"Respond 'None' if the requested information is not explicitly available in the given context."
            )
            context = datapoint['context']
            query = "Extract the measurement for " + f"{measurement} for the entity {item}."
            messages.append((instructions, context, query))

        ctxlm_params = {k: v for k,v in self.sampling_params.items() if k != 'max_tokens'}
        ctxlm_params['max_new_tokens'] = 20
        ctxlm = ContextLM(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            top_k = 10,
            sampling_params=ctxlm_params,
            return_full_output=True,
            verbose = False
        )
        measurement_responses = ctxlm.predict(messages)

        measured_data = []
        for i,response_dict in enumerate(measurement_responses):
            if response_dict['response'].strip().lower() != 'none':
                measured_data.append(
                    self.data[i] | 
                    {
                        'value': response_dict['response']
                        #'context_score': response_dict['context_score'],
                        #'parametric_score': response_dict['parametric_score']
                    } | response_dict['context_score'] | response_dict['parametric_score']
                )

        return measured_data

    
    def _separate_units(self):
        """
        Separates units from the extracted measurements.

        Args:

        Returns:

        """        
        messages = []
        message_data_ids = []
        for i, datapoint in enumerate(self.data):
            item = {k: v for k,v in datapoint.items() if k not in ['context', 'chunk_id', 'measurement', 'value']}
            measurement = datapoint['measurement']
            measurement_val = datapoint['value']
            measurement_description = self.measurement_schema.model_fields[measurement].description
            available_units = self.measurement_schema.model_fields[measurement].json_schema_extra.get('units', None)

            if available_units is not None:
                instructions = (
                    f"You are an expert in data collection and scientific measurements. "
                    f"Given a data point, your task is to separate the value from its unit of measurement. "
                    f"For example, in the data point '5.6 meters', '5.6' is the value and 'meters' is the unit. "
                    f"If the data point does not explicitly include a unit, respond with 'None' as the unit. "
                    f"Otherwise, copy the value and the unit exactly as they appear.\n\n"
                    f"Your response should be formatted as a JSON object with two keys: 'value' and 'unit'. "
                    f"Example: {{'value': 5.6, 'unit': 'meters'}} "
                    f"or {{'value': 5.6, 'unit': 'None'}}"
                )
                context = datapoint['value']
                query = "Separate the value and unit from the given data point."
                prompt = (
                    f"## Instructions:\n{instructions}\n\n## Data Point:\n{context}\n\n## Query:\n{query}"
                )
                messages.append([
                    {"role": "user", "content": prompt}]
                )
                message_data_ids.append(i)

        guided_decoding_params = GuidedDecodingParams(json=BooleanResponse.model_json_schema())
        sampling_params = SamplingParams(
            **self.sampling_params,
            guided_decoding=guided_decoding_params
        )
        responses = self.llm.chat(messages = messages, sampling_params = sampling_params)
        response_texts = [r.outputs[0].text for r in responses]
        response_validated = [
            response_validator(DataPointResponse, r) for r in response_texts
        ]
        
        separated_data = [datapoint for datapoint in self.data]
        for i, resp in enumerate(response_validated):
            separated_data[message_data_ids[i]]['value'] = resp['value']
            separated_data[message_data_ids[i]]['units'] = resp['units']

        return separated_data
    

    def _standardize(self):
        """
        Gives standardized units to the extracted measurements.

        Args:

        Returns:
            
        """        
        messages = []
        message_data_ids = []
        for i, datapoint in enumerate(self.data):
            item = {k: v for k,v in datapoint.items() if k not in ['context', 'chunk_id', 'measurement', 'value']}
            measurement = datapoint['measurement']
            measurement_val = datapoint['value']
            measurement_description = self.measurement_schema.model_fields[measurement].description
            available_units = self.measurement_schema.model_fields[measurement].json_schema_extra.get('units', None)

            if available_units is not None:
                units_str = ', '.join(available_units) + ', other'
                instructions = (
                    f"You are an expert in data collection and scientific measurements. "
                    f"Given a data point, your task is to standardize the format for its unit of measurement by choosing from a list of available options. "
                    f"For example, if the data point is '5.6 meters' and your options are 'm', 'km', 'ft', or 'other', the correct standardized choice would be 'm'. "
                    f"If the data point's unit of measurement is 'None' or null to begin with, simply respond 'None'. "
                    f"Respond with only the standardized unit exactly as it appears in the list of available options. "
                    f"Do not include any additional text or explanation in your response."
                )
                context = f"{{value: {datapoint['value']}, unit: {datapoint['units']}}}"
                query = (
                    f"Determine the best standardized unit for the given data point from among the "
                    f"following choices: {available_units}."
                )
                prompt = (
                    f"## Instructions:\n{instructions}\n\n## Data Point:\n{context}\n\n## Query:\n{query}"
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
        #self.dat = self._separate_units()
        #self.data = self._standardize()

        return self.data
    

    def save(self, filepath: str):
        """
        Saves the measurement data to a csv.

        Args:
            filepath (str): The path to the file where the data will be saved.
        """
        df = pd.DataFrame(self.data)
        df.to_csv(filepath, index=False)



