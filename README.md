# Tiny-Toxic-Detector
All benchmarking and information scripts used in the [Tiny-Toxic-Detector paper}(https://doi.org/10.48550/arXiv.2409.02114).


### Example
For an inference example using the Tiny-Toxic-Detector please see example.py.


### Replication
The main purpose of this repository is full transparency and you should be able to replicate the results in the Tiny-Toxic-Detector paper. 

Please note that results that rely on the base-system are **not** meant to be replicated directly. Rather, you can run the scripts used and compare the model results with those of other models. This should give you roughly the same correlation between the models.

<br>
Informational scripts:

- info_scripts > model_inference_speed_tiny.py | The script to run to measure the inference speed of the Tiny-Toxic-Detector model.
- info_scripts > model_inference_speed.py | The script to run to measure the inference speed of models supported by the transformer framework. This includes the other models used in the Tiny-Toxic-Detector paper.
- info_scripts > model_memory_usage_tiny.py | The script to run to measure the memory usage of the Tiny-Toxic-Detector model.
- info_scripts > model_memory_usage.py | The script to run to measure the memory usage of models supported by the transformer framework. This includes the other models used in the Tiny-Toxic-Detector paper.
<br><br>

Evaluation scripts:
- eval_scripts > toxigen > main.py | The script to run for the majority of transformer models.
- eval_scripts > toxigen > toxicchat.py | The script to run to evaluate the toxicchat model. This is a separate script due to requiring a specific prompting template.
- eval_scripts > toxigen > tiny.py | The script to run to evaluate the tiny-toxic-detector model. This is a separate script due to the custom architecture.
