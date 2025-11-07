import os
import re
from pdf2image import convert_from_path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vllm import LLM, SamplingParams
from .utils import encode_pil_image


class DocumentLM:
    """
    A class to manage and process markdown documents for use with a language model.

    Args:
        model: The vision language model to be used for processing the documents.
        ocr (bool): Whether to use OCR for image-based documents. Defaults to false, 
            in which case the expected input documents are text or markdown files instead of pdfs.
        ocr_prompt (str): The prompt to use for OCR processing when ocr is set to True.
        sampling_params (dict[str, any]): Sampling parameters for the LLM during OCR processing.
        chunk_size (int): The maximum token size of each chunk.
        overlap (int): The number of overlapping characters between consecutive chunks.
        separators (list[str]): A list of custom separators to use when splitting the documents.
    Attributes:
        document_info (list[dict[str, dict]]): A list to store metadata about each document. Each 
            item in the list is a dictionary with a single key-value pair, where the key is 
            the document's filename and the value is another dictionary containing key-value
            metadata, e.g. {'author': 'et al', 'title': 'xyz', 'date': '2028'}.
        documents (list): A list to store the actual document contents.
        chunks (list): A list to store processed chunks of documents.
        chunk_labels (list): A list to store labels or identifiers for each chunk. Specifically, 
            each label is an index referring to the document from which the chunk was derived.
    """
    def __init__(
            self,
            model = None,
            ocr = False,
            ocr_prompt: str = None,
            sampling_params: dict[str, any] = None,
            chunk_size: int = 512,
            overlap: int = 0,
            separators: list[str] = None
        ):
        if ocr and model is None:
            raise ValueError("An OCR-capable model must be provided when ocr is set to True.")
        
        self.model = model
        self.ocr = ocr
        if ocr_prompt is None:
            self.ocr_prompt = "Convert the pdf document to markdown as accurately as possible. Rotate tables if they are presented sideways. Use hash symbols (e.g. #, ##) to indicate headings. Do not start a new line for italic items."
        else:
            self.ocr_prompt = ocr_prompt

        if sampling_params is None:
            self.sampling_params = SamplingParams(
                temperature = 0.1,
                max_tokens = 4096,
            )
        else:
            self.sampling_params = SamplingParams(**sampling_params)

        self.chunk_size = chunk_size
        self.overlap = overlap

        if separators is None:
            separators = [
                "\n\n", "\n"
            ]
        self.separators = separators
        self.filepaths = []
        self.documents = []
        self.chunks = []


    def ocr_read(self, filepaths : list[str]):
        """
        Load pdf documents using OCR, based on provided metadata.

        Args:
            document_info (list[dict[str, dict]]): A list to store metadata about each document. 
                Each item in the list is a dictionary with a single key-value pair, where the key is
                the document's filename and the value is another dictionary containing key-value
                metadata, e.g. {'author': 'et al', 'title': 'xyz', 'date': '2028'}.
        """
        messages = []
        message_paper_ids = []
        for i, filename in enumerate(filepaths):
            images = convert_from_path(filename)
            for page_num, img in enumerate(images):
                if img.mode == "RGBA":
                    img = img.convert("RGB")

                base64_image = encode_pil_image(img)
                image_data_uri = f'data:image/png;base64,{base64_image}'
                message = [
                    {"role": "system", "content": self.ocr_prompt},
                    {
                        "role": "user",
                        "content": [{
                            "type": "image_url",
                            "image_url": {
                            "url": image_data_uri
                            }
                        }],
                    },
                ]
                messages.append(message)
                message_paper_ids.append(i)

        llm = LLM(self.model)
        responses = llm.chat(messages = messages, sampling_params = self.sampling_params)
        response_markdown = [r.outputs[0].text for r in responses]
        documents = [""] * len(filepaths)
        for msg_idx, paper_idx in enumerate(message_paper_ids):
            cleaned_markdown = re.sub(r"^---[\s\S]*?---\s*", "", response_markdown[msg_idx])
            documents[paper_idx] += cleaned_markdown

        self.documents += documents


    def read(self, filepaths : list[str]):
        """
        Load text or markdown documents based on provided metadata.

        Args:
            filepaths (list[str]): A list of file paths to text or markdown documents.
        """
        for filepath in filepaths:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                self.documents.append(content)


    def split(self) -> list[list[str]]:
        """
        Split loaded documents into smaller chunks, using a token-based approach.

        Returns:
            chunks (list[list[str]]): A 2d list of text chunks derived from the documents. Each sublist
                corresponds to a single document, and items within it correspond to small,
                paragraph sized pieces of text.
        """
        for doc_index, document in enumerate(self.documents):
            # Run the recursive splitter
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
                separators = self.separators
            )
            doc_chunks = text_splitter.split_text(document)

            # Remove separators from the text
            for i, sep in enumerate(self.separators):
                doc_chunks = [chunk.replace(sep, "") for chunk in doc_chunks]

            doc_chunks = [t.strip() for t in doc_chunks if len(t.strip()) > 0]
            self.chunks += [doc_chunks]

        return self.chunks
    

    def fit(self, filepaths : list[str]) -> list[list[str]]:
        """
        Load documents based on provided metadata.

        Args:
            filepaths (list[str]): A list of file paths to text, markdown, or pdf documents.

        Returns:
            chunks (list[list[str]]): A 2d list of text chunks derived from the documents. Each sublist
                corresponds to a single document, and items within it correspond to small,
                paragraph sized pieces of text.
        """
        self.filepaths += filepaths
        if self.ocr:
            self.ocr_read(filepaths)
        else:
            self.read(filepaths)
        chunks = self.split()
        return chunks


    def save(self, folderpath: str):
        """
        Save markdown documents to the specified folder.

        Args:
            folderpath (str): The path to the folder where the markdown documents will be saved.
        """
        for doc_index, document in enumerate(self.documents):
            filename = self.filepaths[doc_index]
            base_filename = os.path.basename(filename).replace('.pdf', '.md')
            save_path = os.path.join(folderpath, base_filename)
            with open(save_path, 'w', encoding='utf-8') as file:
                file.write(document)
        


