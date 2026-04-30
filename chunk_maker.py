# The segment section of each clip holds the details about the chunks(smallparts)
import os
import whisper
import json
from deep_translator import GoogleTranslator

model = whisper.load_model("large-v2")

def translate_to_english(text,source_lang):
    if source_lang == "en":
        return text
    try :
        return GoogleTranslator(
            source="auto", 
            target="en",
            ).translate(text)
    except Exception as e :
        print("Translation Error ",e)
        return text

def merge_segement(segments,lang,number,title,min_length = 50):
    chunks = []
    current_group = []
    current_text = ""

    for seg in segments :
        text = seg.get("text","").strip()

        #skip truly empty chunks
        if not text :
            continue

        #accumulate 
        current_group.append(seg)
        current_text += " " + text 

        # once a considerable length is reached make the final chunks
        if len(current_text.strip()) >= min_length:
            #translate
            translated = translate_to_english(current_text.strip(),lang)
            chunks.append({
                "number":number,
                "title" : title,
                "start" : current_group[0]["start"],
                "end" : current_group[-1]["end"],
                "original_text" : current_text.strip(),
                "translated_text" :translated
            })
        # reset the chunks for next
        current_text= ""
        current_group = []

    #handle leftover 
    if current_group and current_text.strip():
        chunks.append({
            "number":number,
            "title" : title,
            "start": current_group[0]["start"],
            "end": current_group[-1]["end"],
            "original_text" : current_text.strip(),
            "translated_text" :translated
        })
    return chunks

def process_audio(filepath,output,number,title):
    print(f"\n Processing : {filepath}")

    result= model.transcribe(
        audio = filepath,
        task = "transcribe",
        language = None
    )

    lang = result.get("language")
    chunks = merge_segement(
        result["segments"],
        lang,
        number,
        title
        )

    with open(output,"w") as f:
        json.dump({
            "number":number,
            "title" : title,
            "language" : lang,
            "chunks" : chunks,
            "full_text" : result["text"]
        },f,indent = 4,ensure_ascii=False)

        print("Saved : ",output)


input_folder = "ex-audio"
output_folder ="Json"

os.makedirs(output_folder,exist_ok=True)

for audio_file in os.listdir(input_folder):
    
    parts = audio_file.split("__")
    number = parts[0] if len(parts) > 0 else "unknown"
    title = parts[1] if len(parts) > 1 else "unknown"
    print(f"Processing : \nFile Number : {number} \nName: {title}")

    file_path = os.path.join(input_folder,audio_file)
    output_file = os.path.join(output_folder,f"{audio_file}.json")

    process_audio(file_path,output_file,number,title)
print("Done")