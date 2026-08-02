import yt_dlp
import os

SAVE_DIR = "data/raw"
os.makedirs(SAVE_DIR, exist_ok=True)

# Need to add a function to download the video from the given URL and save it to the specified output path
def download_video(url, output_path):
    ydl_opts = {
        'outtmpl': output_path,
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def main():
    # Example usage
    video_url = "https://youtu.be/vD0C55-45Zo?si=0OsAgou4ufHl1Ztp"  # Replace with your desired video URL
    output_file = os.path.join(SAVE_DIR, "EVERY PENALTY KICK TAKEN AT THE 2026 FIFA WORLD CUP.mp4")
    download_video(video_url, output_file)
    print(f"Video downloaded and saved to {output_file}")

if __name__ == "__main__":
    main()