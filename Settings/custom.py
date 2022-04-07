#Data
data_path = 'data/train.csv'

#Model use settings
model_name = 'sentence-transformers/all-mpnet-base-v2'
num_res = 10 #How many similar companies do we want to find
char_max = 700 #Limit the number of characters we process for descriptions
top_n = 10 #find the 5 most similar companies

#Model architecture settings
transformer_name = 'bert-base-uncased'
activation_func = nn.Tanh()
batch_size = 10

#Custom settings
words_to_ignore = {"and", "a", "to", "the", ".", ",", "that", "which", "are", "is", "for",
"one", "of", "on", "or", "with", "their"} 