#make vector embeddings out of the extracted texts and make them into pandas dataframe 
# https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
import os
import json
import math
import requests
import pandas as pd

# replace the Nan values 
def clean_embeddings(e):
    return [0 if (isinstance(x,float) and math.isnan(x)) else x for x in e]

def create_embeddings(text_list):
    try :
        r = requests.post(
            "http://localhost:11434/api/embed",
            json={
                "model": "bge-m3",
                "input": text_list
            },
            timeout= 60
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

#list all the jsons
jsons = os.listdir("Json")

#storing all embeddings
my_dict=[]
chunk_id= 0
counter = 0 
batch_size = 20


#get all the json files extracted from audio
for json_files in jsons:

    #open each individual file 
    with open(f"Json/{json_files}") as f:
        content = json.load(f)

    print(f"\n{counter}. Creating embeddings for {json_files} \n")
    counter+=1
    
    texts = []
    valid_indices = []

    #get each individual chunks out of the json file 
    for i,chunk in enumerate(content.get("chunks",[])):
        text = chunk.get("translated_text","") or chunk.get("original_text","")
        text = text.strip()
        if text :
            texts.append(text)
            valid_indices.append(i)
            
    # skip if nothing valid
    if not texts :
        print(f"Skipping {json_files} : (No valid Text)")
        continue

    #Batch Embeddings
    all_embeddings = []
    for i in range(0,len(texts),batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = create_embeddings(batch)

        if batch_embeddings is None:
            print(f"Skipping {json_files} due to batch error")
            break
        all_embeddings.extend(batch_embeddings)

    if all_embeddings is None or len(all_embeddings) != len(texts) :
        print(f"Skipping {json_files} cause : Embedding error")
        continue
    embeddings = all_embeddings
 
    
    # Map embeddings back
    for idx,emb in zip(valid_indices,embeddings):
        chunk = content["chunks"][idx]
        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = clean_embeddings(emb)
        chunk_id +=1
        my_dict.append(chunk)
    

#Build Dataframe
if not my_dict:
    print("No Embeddings Created")
else:
    df = pd.DataFrame.from_records(my_dict)
    print("\nSample data:")
    print(df.head())
    print(f"\n Total Rows : {len(df)}")

