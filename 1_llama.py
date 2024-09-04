

import pandas as pd
dataset = pd.read_csv(r"COPY.csv", encoding = "utf-8")

# Extract abstracts to train on and corresponding titles
article = dataset["Body (cleaned)"]

print(len(article))



# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token='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')
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token='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')




# Llama 2 Tokenizer
import transformers
model_id ="meta-llama/Llama-2-7b-chat-hf"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_id ="meta-llama/Llama-2-7b-chat-hf", trust_remote_code=True)





import transformers
# Our text generator
generator = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    task='text-generation',
    temperature=0.1,
    max_new_tokens=500,
    repetition_penalty=1.1
)




# System prompt describes information given to all conversations
system_prompt = """
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant for labeling topics.
<</SYS>>
"""

# Example prompt demonstrating the output we are looking for
example_prompt = """
I have a topic that contains the following documents:
"केंद्र के तीन कृषि कानूनों के विरोध में महीनों से आंदोलन कर रहे किसानों की संयुक्त समिति ने हरियाणा की मनोहर लाल खट्टर सरकार को आश्वासन दिया है कि वे किसी सरकारी कार्यक्रम में बाधा नहीं पहुंचाएंगे तथा कानून के दायरे में रह कर ही अपनी मांगे रखेंगे।

हिसार मंडलायुक्त चंद्रशेखर ने शुक्रवार को जारी एक बयान में यह जानकारी देते हुए किसान संगठनों की इस संंबध में घोषणा को सराहनीय बताया और कहा है कि किसान नेताओं की सकारात्मक भूमिका के कारण ही सरकार ने गत 16 मई को दर्ज किए गए मामले वापस लेने का निर्णय लिया है। सरकार ने किसानों की मांग को मानते हुए प्रदर्शन के दौरान एक किसान की मौत होने पर उसके एक परिजन को डीसी रेट पर नौकरी देने की अनुमति प्रदान की है। उन्होंने कहा कि प्रशासन किसानों की समस्याओं पर बातचीत के लिए सदैव तत्पर है।

मंडलायुक्त ने किसानों से अपील की है कि आम जनता तथा सर्वहित में वे टीकाकरण तथा स्वास्थ्य कार्यक्रमों में सरकारी अधिकारियों और कर्मचारियों को अपना सहयोग प्रदान करें। देहात में घर-घर सर्वेक्षण बहुत कामयाब रहा है, जिसके कारण कोविड रोगियों को तुरंत इलाज की सुविधा उपलब्ध कराई गई है। 



The topic is described by the following keywords: "Farmers' Protest and Agricultural Reforms, Government-Farmer Dialogue and Negotiations ,  Impact of the Protest on Public Life and Administration"

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.

[/INST] 
"""

# Our main prompt with documents ([DOCUMENTS]) and keywords ([KEYWORDS]) tags
main_prompt = """
[INST]
I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.
[/INST]
"""

prompt = system_prompt + example_prompt + main_prompt






from sentence_transformers import SentenceTransformer

# Pre-calculate embeddings
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-xlm-r-multilingual-v1")
embeddings = embedding_model.encode(article, show_progress_bar=True)




from umap import UMAP
from hdbscan import HDBSCAN

umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom', prediction_data=True)



# Pre-reduce embeddings for visualization purposes
reduced_embeddings = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', random_state=42).fit_transform(embeddings)



from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, TextGeneration

# KeyBERT
keybert = KeyBERTInspired()

# MMR
mmr = MaximalMarginalRelevance(diversity=0.3)

# Text generation with Llama 2
llama2 = TextGeneration(generator, prompt=prompt)

# All representation models
representation_model = {
    "KeyBERT": keybert,
    "Llama2": llama2,
    "MMR": mmr,
}



from bertopic import BERTopic

topic_model = BERTopic(

  # Sub-models
  embedding_model=embedding_model,
  umap_model=umap_model,
  hdbscan_model=hdbscan_model,
  representation_model=representation_model,

  # Hyperparameters
  top_n_words=10,
  verbose=True
)

# Train model
topics, probs = topic_model.fit_transform(article, embeddings)


# Show topics
topic_model.get_topic_info()


