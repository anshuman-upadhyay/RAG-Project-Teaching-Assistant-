import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cs
import joblib
import requests

def create_embeddings(text_list):
    try :
        r = requests.post(
            "http://localhost:11434/api/embed",
            json={
                "model": "bge-m3",
                "input": text_list
            },
            timeout= 180
        )
        r.raise_for_status()
        embedding = r.json()
    except Exception as e :
        print(f"Request Failed : {e}")
        return None
    
    if "embeddings" not in embedding:
        print(f"API Error : {embedding}")
        return None

    return embedding["embeddings"]

def inference(prompt,model):
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout= 180
        )
        response = r.json()
        return response["response"]




#  Load Joblib into this data frame
df= joblib.load("embeddings.joblib")
# get a question asked and create its embeddings 
incoming_query = input("\nAsk a question : ")
query_emb = create_embeddings([incoming_query])
if query_emb is None: 
     print("Failed to embed Query")
     exit()
question_emb = query_emb[0]
# print(question_emb)

#Find similarities of Question embedding with other embeddings stored in the dataframe
similarities = cs(
    np.vstack(df["embedding"]),
    [question_emb]
    ).flatten()
# print(similarities)
top_results = 5
max_idx = similarities.argsort()[::-1][0:top_results]
new_df = df.iloc[max_idx]

#prompt for llm
prompt = f"""
So the user has provided a variety of small videos that are of various topic.They intend to ask questions related to them. You have been given the subtitle chunks containing video title,video number,start and end times in seconds ,along with the original and translated text
{new_df[["title","number","start","end","translated_text"]].to_json(orient = "records")}
--------------------------------------------------
{incoming_query}
User has asked this question related to the video chunks, all you have to do is answer where in a human readable way (no need to mention anything other than the exact answer and all the format is for you) ,Provdie the exact video number and chunks in which the answer is from for better details .If the use asks an unrealted question, tell him that you can only answer the questions that are in context of these videos and not outside them
"""


with open("prompt.txt","w") as f:
    f.write(prompt)
model = "llama3.2:latest"
response = inference(prompt,model)
with open("response.txt","w") as f:
    f.write(response)


# print(new_df[["title","number","original_text"]])

# for idx,items in new_df.iterrows():
#     print(idx,items["title"],items["number"],items["translated_text"])