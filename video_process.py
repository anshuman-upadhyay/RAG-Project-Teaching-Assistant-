# Convert the mp4 videos into mp3 format for whisper
import os
import whisper
import subprocess
files = os.listdir("video")
for file in files:
    tut_no = file.split("[")[1].split("]")[0]
    # print(tut_no)
    file_name = file.split("]")[1].split("Sigma")[0]
    print(tut_no,file_name)
    subprocess.run(["ffmpeg","-i",f"video/{file}",f"ex-audio/{tut_no}_{file_name}.mp3"])