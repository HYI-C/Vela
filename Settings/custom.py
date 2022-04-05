#Data
data_path = 'data/train.csv'

#Model settings
model_name = 'sentence-transformers/all-mpnet-base-v2'
num_res = 10 #How many similar companies do we want to find
char_max = 200 #Limit the number of characters we process for descriptions
top_n = 5 #find the 5 most similar companies
