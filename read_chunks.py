#make vector embeddings out of the extracted texts and make them into pandas dataframe 
import requests
import os
import json
import pandas as pd
def create_embeddings(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )

    embedding = r.json()

    if "embeddings" not in embedding:
        print(embedding)
        return []

    return embedding["embeddings"]
#list all the jsons
jsons = os.listdir("Json")

#storing all embeddings
my_dict=[]
chunk_id= 0

#get all the json files extracted from audio
for json_files in jsons:
    #open each individual file 
    with open(f"Json/{json_files}") as f:
        content = json.load(f)
    #get a list of all the text in each json's embeddings
    embeddings = create_embeddings([x["text"] for x in content["chunks"]])
    print(f"Creating embeddings for {json_files}")
    #get each individual chunks out of the json file 
    for i,chunk in enumerate(content["chunks"]):
        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = embeddings[i]
        chunk_id+=1
        my_dict.append(chunk)
    break
df = pd.DataFrame.from_records(my_dict)
print(df)
