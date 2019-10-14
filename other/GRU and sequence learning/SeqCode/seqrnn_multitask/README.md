# A Recurrent Neural Network Model for Flexible and Adaptive Decision Making based on Sequence Learning
# Source code for the network model and the analyses.

The Python code for the analyses in Zhang et al.(2019)

### Installation
Download the source code into a local directory.

### Set up the path
Open and edit ./seqrnn_multitask/main.py Change variable 'path' to current folder.

### Environment
This code has been tested on python 3.7.3 in linux.

### Generating the datasets
Execute the scripts in ./datagenerator/ to generating the training datasets and validation datasets.
The generated datasets are saved in the folder ./data

### Running the code
1. Edit the configuration files and change rt_shape['data_file'] and rt_shape['validation_data_file'] to the actual file names of the training dataset and validation dataset.
   The configuration files are:
  	./seqrnn_multitask/config.py 		Weather prediction task
  	./seqrnn_multitask/config_mi.py 	Multi-sensory integration task
  	./seqrnn_multitask/config_sure.py 	Post-decision wagering task
  	./seqrnn_multitask/config_st.py 	Two step task

2. Edit the last section of main.py to specify the task.

3. Run main.py

### Datasaving

The trained model is saved in folder ./save_m
The log of the behavioral results and neuronal response is saved in ./log_m 














