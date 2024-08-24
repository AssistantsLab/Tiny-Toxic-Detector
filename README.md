# Tiny-Toxic-Detector
All benchmarking and information scripts used in the Tiny-Toxic-Detector paper.


### Example
For an inference example using the Tiny-Toxic-Detector please see example.py.


### Replication
The main purpose of this repository is full transparency and you should be able to replicate the results in the Tiny-Toxic-Detector paper. 

Please note that results that rely on the base-system are **not** meant to be replicated directly. Rather, you can run the scripts used and compare the model results with those of other models. This should give you roughly the same correlation between the models.

<br>
Informational scripts:

- info_scripts > model_inference_speed_tiny.py | The script to run to measure the inference speed of the Tiny-Toxic-Detector model.
- info_scripts > model_inference_speed.py | The script to run to measure the inference speed of models supported by the transformer framework. This includes the other models used in the Tiny-Toxic-Detector paper.
<br><br>
- eval_scripts > 