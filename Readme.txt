Sample Run:

	step1 : 
		
		Paste the tar file of dataset in the working folder and extract it.
	
	step2 : first run preprocessing.py to get random dataset and attributes.

			python pre_processing.py

	step3 : Now run main.py file to do all the Experiment

			python main.py <Experiment_no>



To generate my result written in report, change path in main.py 

	line number 399 : reviews_arr = get_instances('./files/dataset_train_1.txt')
	line number 400 : test_reviews = get_instances('./files/dataset_test_1.txt')
	line number 401 : vocab_arr = get_attributes('./files/attributes_1.txt')


Used :
	python 2
	OS : Linux 18.04
