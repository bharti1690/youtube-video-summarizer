from pytube import YouTube
from moviepy.editor import VideoFileClip
import whisper
import speech_recognition as sr
from transformers import pipeline

def download_youtube_video(url, output_path='video.mp4'):
    yt = YouTube(url)
    stream = yt.streams.filter(file_extension='mp4').first()
    stream.download(filename=output_path)
    print("Video downloaded successfully")
    return output_path

def extract_audio(video_path, output_audio='audio.wav'):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(output_audio)
    print("Audio extracted successfully")
    return output_audio

def transcribe_audio_whisper(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    return result['text']

def transcribe_audio_google(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return text

def summarize_text(text):
    summarizer = pipeline('summarization', model='t5-small')
    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
    return summary[0]['summary_text']


def youtube_lecture_summarizer(url):
    video_path = download_youtube_video(url)
    audio_path = extract_audio(video_path)
    
    # Choose either Whisper or Google for transcription
    transcript = transcribe_audio_whisper(audio_path)  # Whisper
    # transcript = transcribe_audio_google(audio_path)  # Google
    
    print("Transcription complete")
    
    summary = summarize_text(transcript)
    print("Summary generated")
    
    return summary

if __name__ == "__main__":
    url = input("https://www.youtube.com/watch?v=sTeoEFzVNSc ")
    summary = youtube_lecture_summarizer(url)
    print("\nLecture Summary:\n")
    print(summary)