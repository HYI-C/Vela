from torch import nn

#Data
data_path = 'data/train.csv'

#Model use settings
model_name = 'sentence-transformers/all-mpnet-base-v2'
num_res = 10 #How many similar companies do we want to find
char_max = 256 #Limit the number of characters we process for descriptions
word_max = 20
top_n = 10 #find the 5 most similar companies

#Model architecture settings
transformer_name = 'bert-base-uncased'
activation_func = nn.Tanh()
batch_size = 10

#Custom settings
ignore_words = {"and", "a", "to", "the", ".", ",", "that", "which", "are", "is", "for",
"one", "of", "on", "or", "with", "their", "an"} 

#Validation
sentences1 = ['Grabango is the leader in checkout-free technology for existing, large-scale grocery and convenience stores', 'Grabango is the leader in checkout-free technology for existing, large-scale grocery and convenience stores', 'Grabango is the leader in checkout-free technology for existing, large-scale grocery and convenience stores','Grabango is the leader in checkout-free technology for existing, large-scale grocery and convenience stores','Grabango is the leader in checkout-free technology for existing, large-scale grocery and convenience stores']
sentences2 = ['Zippin is the next generation of checkout-free technology enabling retailers to quickly deploy frictionless shopping in their stores.', 'Standard Cognition provides an autonomous checkout tool that can be installed into retailers’ existing stores.', 'AiFi enables reliable, cost-effective, and contactless autonomous shopping with AI-powered computer vision technology.', 'Moveworks offers an AI platform that revolutionizes how companies support their employees.','Tonkean provides an enterprise no-code process orchestration platform.']
scores = [1, 1, 1, 0.03, 0.05]
eval_threshold = 0.65