'''
	This file is used to create random Train, Test dataset. It also creates 
	random Attributes and write them in the files.
	
'''

import random
import operator
from math import log

# variable declerations
num_positive_reviews = num_negative_reviews = 500
num_vocab = 2500
fraction_validtation = 0.1
num_positive_reviews = int(num_positive_reviews + fraction_validtation*num_positive_reviews)

# function to select the random reviews from the train folder
def choose_random_reviews(num_positive_reviews,filepath):
	
	line_pos_arr = []
	line_neg_arr = []
	for i in range(num_positive_reviews):
		temp = random.randint(1,12500)
		if temp in line_pos_arr:
			while temp in line_pos_arr:
				temp = random.randint(1,12500)
		line_pos_arr.append(temp)

		temp = random.randint(12501,25000)
		if temp in line_neg_arr:
			while temp in line_neg_arr:
				temp = random.randint(12501,25000)
		line_neg_arr.append(temp)

	line_pos_arr.sort()
	line_neg_arr.sort()

	positive_review_arr = []
	negative_review_arr = []

	with open(filepath) as fp:
		line = fp.readline()
		count = 1
		i = 0
		j = 0
		while (line):
			if (i < num_positive_reviews and count == line_pos_arr[i]):
				positive_review_arr.append(line)
				i = i+1
			
			if (j < num_positive_reviews and count == line_neg_arr[j]):
				negative_review_arr.append(line)
				j = j + 1

			line = fp.readline()
			count = count + 1

	return [negative_review_arr,positive_review_arr]

# function to select random attributes from Imdbcocab.txt
def choose_vocab(num_vocab,filepath):
	vocab_arr = []
	with open(filepath) as fp:
		i = 0
		line = fp.readline()
		while(line):
			vocab_arr.append([i,float(line.strip())])
			i = i + 1
			line = fp.readline()


	vocab_arr = sorted(vocab_arr,key=operator.itemgetter(1))
	vocab_pos_arr = [x for x in vocab_arr[-1*num_vocab:]]
	vocab_neg_arr = [x for x in vocab_arr[:num_vocab]]
	return [vocab_neg_arr,vocab_pos_arr]


print("Collecting random features.....")
filepath = r'./aclImdb_v1/aclImdb/train/labeledBow.feat'
reviews_arr = choose_random_reviews(num_positive_reviews,filepath)
print("Random feature selection completed......")

print("\nCollecting random Vocab.....")
filepath = r'./aclImdb_v1/aclImdb/imdbEr.txt'
vocab_arr = choose_vocab(num_vocab,filepath)
print("Random vocab selection completed.....")

filepath = r'./aclImdb_v1/aclImdb/test/labeledBow.feat'
test_reviews = choose_random_reviews(num_negative_reviews,filepath)


positive_review_arr = reviews_arr[1]
negative_review_arr = reviews_arr[0]


vocab_pos_arr = vocab_arr[1]
vocab_neg_arr = vocab_arr[0]
vocab_arr = vocab_arr[1] + vocab_arr[0]

vocab_map = {}
for i in range(len(vocab_pos_arr)):
	vocab_map[vocab_pos_arr[i][0]] = vocab_pos_arr[i][1]

for i in range(len(vocab_neg_arr)):
	vocab_map[vocab_neg_arr[i][0]] = vocab_neg_arr[i][1]



# modifying the positive and negative review array to store only the index present in the vocab array
def modify_review_arr(arr,dict_vocab):
	for i in range(len(arr)):
		temp = arr[i]
		temp_arr = temp.split()
		temp_line = [int(temp_arr[0])]
		temp_dict = {}
		for j in range(1,len(temp_arr)):
			temp2 = temp_arr[j]
			temp2_arr = temp2.split(':')
			if int(temp2_arr[0]) in dict_vocab:
				temp_dict[int(temp2_arr[0])] = int(temp2_arr[1])
			
		temp_line.append(temp_dict)

		arr[i] = temp_line
	return arr

test_reviews_pos =  modify_review_arr(test_reviews[0],vocab_map)
test_reviews_neg =  modify_review_arr(test_reviews[1],vocab_map)
test_reviews = test_reviews_neg + test_reviews_pos

positive_review_arr = modify_review_arr(positive_review_arr,vocab_map)
negative_review_arr = modify_review_arr(negative_review_arr,vocab_map)
reviews_arr = positive_review_arr + negative_review_arr


# this part of code is used to write the selected random data to dataset_train.txt
fo = open("dataset_train.txt", "w")
for i in range(len(reviews_arr)):
	fo.write(str(reviews_arr[i][0]) + " ")
	for j in reviews_arr[i][1]:
		fo.write(str(j) + ":" + str(reviews_arr[i][1][j]) + " ")
	fo.write("\n")

# this part of code is used to write the selected random test data to dataset_test.txt
fo = open("dataset_test.txt", "w")
for i in range(len(test_reviews)):
	fo.write(str(test_reviews[i][0]) + " ")
	for j in test_reviews[i][1]:
		fo.write(str(j) + ":" + str(test_reviews[i][1][j]) + " ")
	# fo.write(str(reviews_arr[i]))
	fo.write("\n")



len_validation_arr = fraction_validtation*num_negative_reviews

validation_arr = reviews_arr[0:int(len_validation_arr)]
temp = reviews_arr[(len(reviews_arr)-int(len_validation_arr)):]
validation_arr = validation_arr + temp

reviews_arr = reviews_arr[int(len_validation_arr):(len(reviews_arr)-int(len_validation_arr))]


# this part of code is used to write the selected random attributes to attributes.txt
fo = open("attributes.txt", "w")
for i in range(len(vocab_arr)):
	fo.write(str(vocab_arr[i][0])+ " " + str(vocab_arr[i][1]))
	fo.write("\n")

