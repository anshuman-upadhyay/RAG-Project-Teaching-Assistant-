# How to use this RAG AI Teaching assistant on your own data
## Step 1 : Collect Video
Move all the required videos files to video folder
## Step 2 : Convert To mp3
Convert all the videos into mp3 by running video_to_mp3

## Step 3 : Convert To json
Convert all the mp3 into .json files by running mp3_to_json

## Step 4 : Convert To json files to Vectors
Use the files in Json folder to convert .json files into a dataframe and save then as joblib pickle using json_Embeddings
 
## Step 5 :Prompt generation and feeding to LLm
Read the joblib file and load it into memory. Then Create a relevant promt as per the user query and feed it to the LLM 



