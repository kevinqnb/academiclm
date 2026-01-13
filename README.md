# AcademicLM :microscope: :books:

**Parse and analyze scientific research papers with large language models.**

*NOTE:* This project is a work in progress, and only a portion of it is shared here for the purpose of communicating my work. I kindly ask that you please be respectful of the content, for now it is shared for viewing and discussion only.

This library implements a system for extracting insights from scientific papers (which are in the form of pdfs) using large language models.
Specifically, we apply local and open source LLMs towards organized tasks for:
* Document OCR: translating pdf images into markdown, and splitting into paragraph sized chunks.
* Document extraction: systematically collecting data points from chunks of markdown text. 
* Hallucination detection

Our focus is on using small, local models for OCR and text generation tasks, and this library is designed to be compatible with any such model of your choosing. 

### Hallucination Detection
To detect hallucinations whilst extracting data, we implement methods from mechanistic 
interpretability to analyze a model's internal features. Specifically, we 
use the scoring mechanisms defined in the following work. 

**Sun, Zhongxiang, et al. "ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation via Mechanistic Interpretability." ICLR. 2025.**

This allows for computation of scores across model layers and attention heads, of which 
some may be more indicative of hallucination behavior than others. 

<table>
  <tr>
    <td align="center">
      <img src="data/figures/context_truth.png" width="420" /><br/>
      <sub>True Extraction</sub>
    </td>
    <td align="center">
      <img src="data/figures/context_hallucinated.png" width="420" /><br/>
      <sub>Hallucinated Extraction</sub>
    </td>
  </tr>
</table>



### Basic Usage
We include an example for extracting forest carbon mass measurements in 
`examples/forests.py`.

## License

Copyright (c) 2025 [Kevin Quinn]. All rights reserved.

This repository and its contents are provided for viewing purposes only.
No part of this work may be reproduced, distributed, or used in any form
without the express written permission of the author.

