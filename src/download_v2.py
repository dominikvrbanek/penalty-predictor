# download v2 - trying to fix ERROR 403 Forbidden and ERROR 429 Too Many Requests
# Attempt with extracting cookies from the browser (Chrome) using yt-dlp's cookiesfrombrowser option.
# Or try to use ffmpeg directly with the URL, but that might not work for YouTube due to their restrictions.
import yt_dlp
import os

SAVE_DIR = "data/raw"
os.makedirs(SAVE_DIR, exist_ok=True)

# Need to add a function to download the video from the given URL and save it to the specified output path
def download_video(url, output_path):
    ydl_opts = {
        'outtmpl': output_path,
        'cookiefile': 'cookies.txt',
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
        # Fix for mid-download 403s
        'http_chunk_size': 10485760,  # download in 10MB chunks
        'retries': 10,
        'fragment_retries': 10,
        'continuedl': True,
        'extractor_args': {'youtube': {'player_client': ['ios']}},
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

'''
def download_video(url, output_path):
    ydl_opts = {
        'outtmpl': output_path,
        'cookiefile': 'cookies.txt',
        'format': 'bestvideo[height<=1080][ext=mp4]+bestaudio/best',
        'merge_output_format': 'mp4',
        'http_chunk_size': 10485760,
        'retries': 10,
        'fragment_retries': 10,
        'continuedl': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
'''

def main():
    # Example usage
    video_url = "https://youtu.be/uOdIYcBpy78?si=Wu2TZX46KlGGJ1nl"  # Replace with your desired video URL
    output_file = os.path.join(SAVE_DIR, "Every Penalty Save of the Season Top Saves Serie A 2023-24.mp4")  # Change the output file name as needed
    download_video(video_url, output_file)
    print(f"Video downloaded and saved to {output_file}")

if __name__ == "__main__":
    main()