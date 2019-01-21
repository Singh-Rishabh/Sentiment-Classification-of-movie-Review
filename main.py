'''
	This is the main file and used to build tree and 
	do all the experiment.
'''

# Import Statements
import random
import operator
import subprocess
from math import log
import sys
from copy import deepcopy
# from pre_processing import num_positive_reviews,num_negative_reviews,num_vocab,fraction_validtation

len_arg = len(sys.argv)
if (len_arg != 2):
	print ("Usage: python main.py <Experiment_no> ..... !!!!")
	exit()

Experiment_no = int(sys.argv[1])

# Variables 
num_positive_reviews = num_negative_reviews = 500
num_vocab = 2500
fraction_validtation = 0.1
num_positive_reviews = int(num_positive_reviews + fraction_validtation*num_positive_reviews)


# function to read attributes from attributes.txt file
def get_attributes(filepath):
	vocab_arr = []
	with open(filepath) as fp:
		line = fp.readline()
		while (line):
			temp_arr = line.split()
			temp_arr[0] = int(temp_arr[0])
			temp_arr[1] = float(temp_arr[1])
			vocab_arr.append(temp_arr)
			line = fp.readline()

	return vocab_arr

# function to build the training dataset_arr and test dataset array.
def get_instances(filepath):
	reviews_arr = []
	with open(filepath) as fp:
		line = fp.readline()
		while (line):
			temp_arr = line.split()
			instance = [int(temp_arr[0])]
			temp_dict = {}
			for j in range(1,len(temp_arr)):
				temp2 = temp_arr[j]
				temp2_arr = temp2.split(':')
				temp_dict[int(temp2_arr[0])] = int(temp2_arr[1])
			instance.append(temp_dict)
			reviews_arr.append(instance)
			line = fp.readline()

	return reviews_arr



# Node class
class Node:
    def __init__(self, data=-1):
		self.left = None
		self.right = None
		self.index = data
		self.numPos = 0
		self.numNeg = 0
		self.class_name = None
		self.is_leaf = 0


# utility function to count number of nodes 
def count_nodes(root):
	if (root == None):
		return 0
	# count = count + 1
	x = count_nodes(root.left)
	y = count_nodes(root.right)
	return x+ y + 1

# utility function to get height of tree.
def height(node):
    if node is None:
        return 0
    else :
        lheight = height(node.left)
        rheight = height(node.right)
 
        if lheight > rheight :
            return lheight+1
        else:
            return rheight+1

# utility function to print level order transversal of tree
def printLevelOrder(root):
    h = height(root)
    for i in range(1, h+1):
        printGivenLevel(root, i)
        print()
 
 
def printGivenLevel(root , level):
    if root is None:
        return
    if level == 1:
        print (root.index,root.numPos,root.numNeg,root.is_leaf,root.class_name),
    elif level > 1 :
        printGivenLevel(root.left , level-1)
        printGivenLevel(root.right , level-1)


# function to count number of terminal node in tree
def count_terminal_node(root):
    num_terminal_node = 0
    most_frequent_split = {}
    if root is None:
        return num_terminal_node,most_frequent_split
     
    queue = []
 	
    queue.append(root)
 
    while(len(queue) > 0):
		
        if queue[0].index != -1:
	        if queue[0].index in most_frequent_split:
	        	most_frequent_split[queue[0].index] += 1
	        else:
	        	most_frequent_split[queue[0].index] = 1

        if (queue[0].is_leaf == 1):
        	num_terminal_node += 1
        node = queue.pop(0)

        if node.left is not None:
            queue.append(node.left)
 
        if node.right is not None:
            queue.append(node.right)

    maxx = -1
    max_split_index = -1
    for key, value in most_frequent_split.iteritems():
    	if (value > maxx):
    		maxx = value
    		max_split_index = key

    return num_terminal_node,max_split_index,maxx


# utility function to count positve class in decision tree
def count_postive_review(dataset):
	count = 0
	for i in range(len(dataset)):
		if (dataset[i][0] >= 7 ):
			count = count + 1
	return count 


def is_present(dataset,index):
	is_present_arr = []
	for i in range(len(dataset)):
		if index in dataset[i][1]:
			is_present_arr.append(dataset[i])
	return is_present_arr

# utility funtion to calculate the maximum information gain
def max_ig(dataset,vocab_arr):
	ig_arr = []
	num_pos_reviews = count_postive_review(dataset)
	num_neg_reviews = len(dataset) - num_pos_reviews
	if (num_pos_reviews + num_neg_reviews != 0):
		p_pos = float(num_pos_reviews)/(num_pos_reviews+ num_neg_reviews)
		p_neg = float(num_neg_reviews)/(num_pos_reviews+ num_neg_reviews)
	else:
		p_pos = 0
		p_neg = 0

	if (p_pos == 0 or p_neg == 0):
		entropy = 0
	else:
		entropy = -1.0*p_pos*log(p_pos,2) - p_neg*log(p_neg,2)
	maxx = -100
	index_max = -1
	for i in range(len(vocab_arr)):
		is_present_arr = is_present(dataset,vocab_arr[i][0])
		not_present_arr = [item for item in dataset if item not in is_present_arr]

		num_pos_reviews_in_presesnt_branch = count_postive_review(is_present_arr)
		num_neg_reviews_in_presesnt_branch = len(is_present_arr) - num_pos_reviews_in_presesnt_branch


		num_pos_reviews_in_not_presesnt_branch = count_postive_review(not_present_arr)
		num_neg_reviews_in_not_presesnt_branch = len(not_present_arr) - num_pos_reviews_in_not_presesnt_branch

		if (num_pos_reviews_in_not_presesnt_branch + num_neg_reviews_in_not_presesnt_branch != 0) : 
			p_pos1 = float(num_pos_reviews_in_not_presesnt_branch)/(num_pos_reviews_in_not_presesnt_branch + num_neg_reviews_in_not_presesnt_branch)
			p_neg1 = float(num_neg_reviews_in_not_presesnt_branch)/(num_pos_reviews_in_not_presesnt_branch + num_neg_reviews_in_not_presesnt_branch)
		else:
			p_pos1 = 0
			p_neg1 = 0
		if (num_pos_reviews_in_presesnt_branch + num_neg_reviews_in_presesnt_branch != 0):
			p_pos2 = float(num_pos_reviews_in_presesnt_branch)/(num_pos_reviews_in_presesnt_branch + num_neg_reviews_in_presesnt_branch)
			p_neg2 = float(num_neg_reviews_in_presesnt_branch)/(num_pos_reviews_in_presesnt_branch + num_neg_reviews_in_presesnt_branch)
		else:
			p_pos2 = 0
			p_neg2 = 0
		if (p_pos1 == 0 or p_neg1 == 0):
			e1 = 0
		else:
			e1 = -1.0*p_pos1*log(p_pos1,2) - p_neg1*log(p_neg1,2)

		if (p_pos2 == 0 or p_neg2 == 0):
			e2 = 0
		else:
			e2 = -1.0*p_pos2*log(p_pos2,2) - p_neg2*log(p_neg2,2)
		ig = entropy - float(len(not_present_arr))/len(dataset)*e1 - float(len(is_present_arr))/len(dataset)*e2
		ig_arr.append(ig)
		if (ig > maxx):
			maxx = ig
			index_max = vocab_arr[i][0]
			temp_arr = is_present_arr

	out_index = -1
	if (maxx == 0):
		out_index = -1
	else:
		out_index = index_max
	return out_index,maxx


# function to build the decision tree
def build_tree(dataset,vocab_arr,max_height):
	root = Node()
	
	num_pos_reviews = count_postive_review(dataset)
	num_neg_reviews = len(dataset) - num_pos_reviews
	if (max_height == 0):
		if (num_pos_reviews> num_neg_reviews):
			root.class_name = '1'
			root.numPos = num_pos_reviews
			root.numNeg = num_neg_reviews
			root.is_leaf = 1
		else:
			root.numPos = num_pos_reviews
			root.numNeg = num_neg_reviews
			root.is_leaf = 1
			root.class_name = '-1'
		
		return root

	if (num_pos_reviews == 0):
		root.class_name = '-1'
		root.numPos = 0
		root.numNeg = num_neg_reviews
		root.is_leaf = 1
		return root

	if (num_neg_reviews == 0):
		root.class_name = '1'
		root.numPos = num_pos_reviews
		root.numNeg = 0
		root.is_leaf = 1
		return root

	index,maxx = max_ig(dataset,vocab_arr)
	root.numPos = num_pos_reviews
	root.numNeg = num_neg_reviews
	# vocab_arr.remove(vocab_arr[index])
	is_present_arr = is_present(dataset,index)
	not_present_arr = [x for x in dataset if x not in is_present_arr]

	root.index = index
	if (index == -1):
		if (num_pos_reviews> num_neg_reviews):
			root.class_name = '1'
			root.is_leaf = 1
		else:
			root.is_leaf = 1
			root.class_name = '-1'

	else:
		root.left = build_tree(not_present_arr,vocab_arr,max_height-1)
		root.right = build_tree(is_present_arr,vocab_arr,max_height-1)
	
	return root

# utility function to calculate output class label
def get_outout(dict_index,root):
	if (root.index == -1 ):
		return root.class_name
	if root.index in dict_index:
		return get_outout(dict_index,root.right)
	elif root.index not in dict_index:
		return get_outout (dict_index,root.left)

# function to calculate accuracy
def calc_accuracy(arr,root):
	sum = 0
	for i in range(len(arr)):
		x = get_outout(arr[i][1],root)
		
		y = arr[i][0]
		if (y >= 7):
			y = '1'
		else:
			y = '-1' 

		if (x == y):
			sum += 1
	return (float(sum)*100.0/len(arr))


# utility function to do pruning on the tree.
def pruning_util(root,node,validation_arr):
    global global_accuracy
  
    if (node == None):
        return
    if (node.is_leaf == 1):
        return
    # print("node is not none")
    right_copy = deepcopy(node.right)
    left_copy = deepcopy(node.left)
    temp_is_leaf = deepcopy(node.is_leaf)
    temp_class_name = deepcopy(node.class_name)
    temp_index = deepcopy(node.index)


    node.left = None
    node.right = None
    node.is_leaf = 1
    if(node.numPos>node.numNeg):
    	node.class_name = '1'
    else :
    	node.class_name = '-1'
    node.index = -1

    temp_acc = calc_accuracy(validation_arr,root)
    if (temp_acc > global_accuracy):
    	global_accuracy = temp_acc
    	print("\n\n=======>>increased accruacy to " + str(global_accuracy))
        print("=======>>Number of nodes " + str(count_nodes(root)) )
        print("=======>>height of tree = " + str(height(root)))

        return
    else :
    	node.left = left_copy
    	node.right = right_copy
    	node.leaf = temp_is_leaf
    	node.class_name = temp_class_name
    	node.index = temp_index
    	if (node.left != None):
    		pruning_util(root , node.left , validation_arr )
    	if (node.right != None):
    		pruning_util(root , node.right , validation_arr )

def pruning(root,node,validation_arr):
	global global_accuracy 
	global acc_original
	pruning_util(root,node,validation_arr)
	print(global_accuracy,acc_original)
	while (global_accuracy > acc_original):
		# print("iteration next time")
		acc_original = global_accuracy
		pruning_util(root,node,validation_arr)

# function to calculate accuracy of random forest
def calc_accuracy_forest(arr,root_arr):
	sum = 0
	print(len(root_arr))
	for i in range(len(arr)):
		count_pos = 0
		for j in range(len(root_arr)):
			x = get_outout(arr[i][1],root_arr[j])
			if (x == '1'):
				count_pos += 1
		if (count_pos > len(root_arr)/2 ):
			x = '1'
		else :
			x = '-1'
		
		y = arr[i][0]
		if (y >= 7):
			y = '1'
		else:
			y = '-1' 

		if (x == y):
			sum += 1
	return (float(sum)*100.0/len(arr))


#  change path here *********************
reviews_arr = get_instances('./dataset_train.txt')
test_reviews = get_instances('./dataset_test.txt')
vocab_arr = get_attributes('./attributes.txt')

len_validation_arr = fraction_validtation*num_negative_reviews

validation_arr = reviews_arr[0:int(len_validation_arr)]
temp = reviews_arr[(len(reviews_arr)-int(len_validation_arr)):]
validation_arr = validation_arr + temp

reviews_arr = reviews_arr[int(len_validation_arr):(len(reviews_arr)-int(len_validation_arr))]


print("\nBuilding the Decision Tree......")

root = build_tree(reviews_arr,vocab_arr,-1)
tree_height = height(root)
print("height of tree = " + str(tree_height))
print("number of terminal node = "+ str(count_terminal_node(root)[0]) )
print ("--------Number of nodes " + str(count_nodes(root)) )
print("Decision Tree Building Completed......")

acc_original = calc_accuracy(test_reviews,root)
acc_original_traning = calc_accuracy(reviews_arr,root)
acc_original_copy = acc_original
global_accuracy = acc_original

print("\n----->>Accuracies of the tree on traning dataset and test dataset are \n\t(%f, %f)" %(acc_original_traning,acc_original))

if (Experiment_no == 1):
	print("Dataset already build...Now Run experiment 2")

	
if (Experiment_no == 2):
	max_height1 = int(tree_height - tree_height/100.0*10)
	max_height2 = int(tree_height - tree_height/100.0*25)
	max_height3 = int(tree_height - tree_height/100.0*45)
	max_height4 = int(tree_height - tree_height/100.0*60)
	max_height5 = int(tree_height - tree_height/100.0*70)

	early_stopping_root1 = None
	early_stopping_root2 = None
	early_stopping_root3 = None
	early_stopping_root4 = None
	early_stopping_root5 = None

	print("\nBuilding the Early stopped Decision Tree of height 90,75,55,40,30 percentage of actual height......")

	early_stopping_root1 = build_tree(reviews_arr,vocab_arr,max_height1)
	early_stopping_root2 = build_tree(reviews_arr,vocab_arr,max_height2)
	early_stopping_root3 = build_tree(reviews_arr,vocab_arr,max_height3)
	early_stopping_root4 = build_tree(reviews_arr,vocab_arr,max_height4)
	early_stopping_root5 = build_tree(reviews_arr,vocab_arr,max_height5)

	print("heights of early stopped tree")
	print(tree_height, height(early_stopping_root1), height(early_stopping_root2), height(early_stopping_root3), height(early_stopping_root4),height(early_stopping_root5))

	es_acc1 = calc_accuracy(test_reviews,early_stopping_root1)
	es_acc2 = calc_accuracy(test_reviews,early_stopping_root2)
	es_acc3 = calc_accuracy(test_reviews,early_stopping_root3)
	es_acc4 = calc_accuracy(test_reviews,early_stopping_root4)
	es_acc5 = calc_accuracy(test_reviews,early_stopping_root5)

	es_acc1_traning = calc_accuracy(reviews_arr,early_stopping_root1)
	es_acc2_traning = calc_accuracy(reviews_arr,early_stopping_root2)
	es_acc3_traning = calc_accuracy(reviews_arr,early_stopping_root3)
	es_acc4_traning = calc_accuracy(reviews_arr,early_stopping_root4)
	es_acc5_traning = calc_accuracy(reviews_arr,early_stopping_root5)

	print("\n----->>Accuracies of the early stopped tree on traning dataset \n\t(%f, %f, %f, %f, %f)" %(es_acc1_traning,es_acc2_traning,es_acc3_traning,es_acc4_traning,es_acc5_traning))
	print("\n----->>Accuracies of the early stopped tree on test dataset \n\t(%f, %f, %f, %f, %f)" %(es_acc1,es_acc2,es_acc3,es_acc4,es_acc5))

	num_terminal_node,most_frequent_split,max_val = count_terminal_node(root)

	num_terminal_node1,most_frequent_split1,max_val1 = count_terminal_node(early_stopping_root1)
	num_terminal_node2,most_frequent_split2,max_val2 = count_terminal_node(early_stopping_root2)
	num_terminal_node3,most_frequent_split3,max_val3 = count_terminal_node(early_stopping_root3)
	num_terminal_node4,most_frequent_split4,max_val4 = count_terminal_node(early_stopping_root4)
	num_terminal_node5,most_frequent_split5,max_val5 = count_terminal_node(early_stopping_root5)

	print("\n----->>Number of terminal node on the pure Decision tree is %d and \n\tattribute that is most frequently used to split the node is %d which is used %d times" %(num_terminal_node,most_frequent_split,max_val))
	print("\n----->>Number of terminal nodes on early stopped tree are %d, %d, %d, %d, %d" %(num_terminal_node1,num_terminal_node2,num_terminal_node3,num_terminal_node4,num_terminal_node5))
	print("\n----->>Most frequently used node to split early stopped tree are %d, %d, %d, %d, %d and they are used %d, %d, %d, %d, %d times respectively." %(most_frequent_split1,most_frequent_split2,most_frequent_split3,most_frequent_split4,most_frequent_split5,max_val1,max_val2,max_val3,max_val4,max_val5) )

elif(Experiment_no == 3):
	print("\nAdding noise to the traning dataset....")
	noise_arr1 = deepcopy(reviews_arr)
	noise_arr2 = deepcopy(reviews_arr)
	noise_arr3 = deepcopy(reviews_arr)
	noise_arr4 = deepcopy(reviews_arr)

	for i in range(int(0.5*len(noise_arr1)/100)):
		temp = random.randint(0,len(noise_arr1)-1)
		if (noise_arr1[temp][0] >= 7):
			noise_arr1[temp][0] = 3
		else:
			noise_arr1[temp][0] = 8

	for i in range(1*len(noise_arr2)/100):
		temp = random.randint(0,len(noise_arr2)-1)
		if (noise_arr2[temp][0] >= 7):
			noise_arr2[temp][0] = 3
		else:
			noise_arr2[temp][0] = 8

	for i in range(5*len(noise_arr3)/100):
		temp = random.randint(0,len(noise_arr3)-1)
		if (noise_arr3[temp][0] >= 7):
			noise_arr3[temp][0] = 3
		else:
			noise_arr3[temp][0] = 8



	for i in range(10*len(noise_arr4)/100):
		temp = random.randint(0,len(noise_arr4)-1)
		if (noise_arr4[temp][0] >= 7):
			noise_arr4[temp][0] = 3
		else:
			noise_arr4[temp][0] = 8

	noise_root1 = build_tree(noise_arr1,vocab_arr,-1)
	noise_root2 = build_tree(noise_arr2,vocab_arr,-1)
	noise_root3 = build_tree(noise_arr3,vocab_arr,-1)
	noise_root4 = build_tree(noise_arr4,vocab_arr,-1)

	
	noise_acc1 = calc_accuracy(test_reviews,noise_root1)
	noise_acc2 = calc_accuracy(test_reviews,noise_root2)
	noise_acc3 = calc_accuracy(test_reviews,noise_root3)
	noise_acc4 = calc_accuracy(test_reviews,noise_root4)
	num_terminal_node1,most_frequent_split1,max_val1 = count_terminal_node(noise_root1)
	num_terminal_node2,most_frequent_split2,max_val2 = count_terminal_node(noise_root2)
	num_terminal_node3,most_frequent_split3,max_val3 = count_terminal_node(noise_root3)
	num_terminal_node4,most_frequent_split4,max_val4 = count_terminal_node(noise_root4)

	print("\n----->>Accuracy after adding noise in the traning dataset\n\t(%f, %f, %f, %f)" %(noise_acc1,noise_acc2,noise_acc3,noise_acc4))
	print("\n----->>Number of terminal nodes after adding noise in traning dataset\n\t(%d, %d, %d, %d)" %(num_terminal_node1,num_terminal_node2,num_terminal_node3,num_terminal_node4))
	print("\n----->>Number of nodes after adding noise in traning dataset\n\t(%d, %d, %d, %d)" %(count_nodes(noise_root1),count_nodes(noise_root2), count_nodes(noise_root3),count_nodes(noise_root4)))
	print("\n----->>Height of trees after adding noise \n\t(%d, %d, %d, %d)" %( height(noise_root1), height(noise_root2),height(noise_root3),height(noise_root4) ) )

elif (Experiment_no == 4):
	print("\nPerforming post-pruning on Decision tree.....")

	pruning(root,root,test_reviews)
	print("\n----->>Accuracy after Performing post-pruning on Decision tree: %f" %(global_accuracy) )
	# print("\n----->>Number of nodes after performing pruning ")
	# print(global_accuracy)

elif (Experiment_no == 5):
	num_trees_in_forest = 3
	num_vocab_in_forest = 2000
	forest_vocab_carr = []
	for i in range(num_trees_in_forest):
		forest_vocab_arr = []
		for i in range(num_vocab_in_forest):
			temp = random.randint(0,len(vocab_arr)-1)
			while vocab_arr[temp] in forest_vocab_arr:
				temp = random.randint(0,len(vocab_arr)-1)
			forest_vocab_arr.append(vocab_arr[temp])
		forest_vocab_carr.append(forest_vocab_arr)

	print("\nBuilding random forest with tree count = " + str(num_trees_in_forest))

	forest_root_arr = []
	for i in range(num_trees_in_forest):
		root_temp = build_tree(reviews_arr,forest_vocab_carr[i],-1)
		forest_root_arr.append(root_temp)

	print("\nRandom forst construction completed....")

	
	acc = calc_accuracy_forest(test_reviews,forest_root_arr)
	print("\n----->>Accuracy after Random Forest construction is %f" %(acc))


