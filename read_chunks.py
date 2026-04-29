#make vector embeddings out of the extracted texts and make them into pandas dataframe 
import requests
def create_embeddings(text):
    r= requests.post("http://localhost:11434/api/embeddings",
              json = {
                  "model" : "bge-m3",
                  "prompt":"Watasiwa Kami"
              }
              )
    embedding = r.json()['embedding']
    return embedding
print(create_embeddings("Anshuman the goat"))