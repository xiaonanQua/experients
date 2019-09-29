# A Recurrent Neural Network Model for Flexible and Adaptive Decision Making based on Sequence Learning
# Source code for the network model and the analyses.

The Python code for the analyses in Zhang et al.(2019)

### Installation
Download the source code into a local directory.

### Set up the path
Open and edit ./seqrnn_multitask/main.py Change variable 'path' to current folder.

### Environment
This code has been tested on python 3.7.3 in linux. All the required packages are listed in the file 'requirements.txt'

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

### Data analysis

The analysis code is in the folder ./holmes_m

	** probability reasoning task **
		- 'bhv_check.py': figure 2, plotting the psychometric curve and the reaction time distribution 
		- 'subweight.py': figure 2, plotting the subjective weight of each shape
		- 'neuron_PSTH.py' figure3, plotting the psth of neurons with different selectivities
		- 'variance.py': figure3, the variance of units' responses
		- 'pred_shape.py': figure 5, plotting the prediction of output units on the appearance of shapes
		
		
		- 'find_wh.py': find the when/which units and save the result in a .yaml file
		- 'clustering.py': figure 4, graph analysis
		- 'bhv_check.py': figure 4, plotting the effect of when/which lesion
		- 'sa_trade.py': figure 4, showing the how the accuracy and the reaction time change with the when units inactivation

	** multisensory integration task **
		-'MI_ana.py': contains all the analyses for the multisensory integration task


	** Post-decision wagering task **
		-'Sure_ana.py': contains all the analyses for the post-decision wagering task

	** Two-step task **
		-'ST_ana.py': contains all the analyses for the two-step task

### Data analysis

The figures are stored in ./figs





