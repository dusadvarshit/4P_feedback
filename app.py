import streamlit as st
import intelligent_feedback

import random
import string
import os

try:
    os.mkdir("tmp")
except:
    pass

def file_creator(audio_file):
    characters = string.ascii_letters + string.digits + string.punctuation
    
    # Generate the random string
    file_name = ''.join(random.choice(characters) for _ in range(15)).replace("/","-")+".wav"
    audio_bytes = audio_file.read()

    with open(os.path.join("tmp", file_name), "wb") as f:
        f.write(audio_bytes)

    return "tmp/"+file_name

def file_destroyer(file_location):
    try:
        os.remove(file_location)
    except:
        pass

st.title('Intelligent Audio Feedback')

# Display image
st.subheader('Upload an audio file')
st.write('Only .wav allowed')
audio_file = st.file_uploader(label="Upload your picture", type=['.wav'])

if audio_file!= None:
    st.audio(audio_file, format="audio/wav")
    
    file_location = file_creator(audio_file)

    results = intelligent_feedback.main(file_location)

    
    st.write('Pause Score:', results['Pause']["Score"])
    st.write('Pause feedback:', " ".join(results['Pause']["Text"]))
    
    st.write('Pace Score:', results['Pace']["Score"])
    st.write('Pace feedback:', " ".join(results['Pace']["Text"]))

    st.write('Power Score:', results['Power']["Score"])
    st.write('Power feedback:', " ".join(results['Power']["Text"]))
    
    st.write('Pitch Score:', results['Pitch']["Score"])
    st.write('Pitch feedback:', " ".join(results['Pitch']["Text"]))

    file_destroyer(file_location)



    

