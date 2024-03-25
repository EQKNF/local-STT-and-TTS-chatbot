import pyaudio  

import wave  

audio_path = "misc/response.wav"

def play_audio(audio_path): 
    chunk = 1024  
    

    f = wave.open(audio_path,"rb")  

    p = pyaudio.PyAudio()  
 
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                    channels = f.getnchannels(),  
                    rate = f.getframerate(),  
                    output = True)  
 
    data = f.readframes(chunk)  
     
    while data:  
        stream.write(data)  
        data = f.readframes(chunk)  
    
    stream.stop_stream()  
    stream.close()  
    p.terminate()


if __name__ == "__main__": 
    play_audio(audio_path)
    