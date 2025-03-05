import requests
import zipfile
import io
import os


def get_movie_words(url):
    # Download the file
    response = requests.get(url, stream=True)

    # Save the file for debugging
    zip_filename = "downloaded_file.zip"
    with open(zip_filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("File downloaded successfully.")

    # Try opening the ZIP file
    try:
        with zipfile.ZipFile(zip_filename, "r") as zip_file:
            extract_path = "extracted_srt"
            zip_file.extractall(extract_path)

            # Find the SRT file inside the extracted contents
            for file_name in os.listdir(extract_path):
                if file_name.endswith(".srt"):
                    srt_path = os.path.join(extract_path, file_name)
                    print("Found SRT file:", srt_path)

                    # Read the SRT file
                    with open(srt_path, "r", encoding="utf-8") as srt_file:
                        return srt_file.read()

    except zipfile.BadZipFile:
        print("Error: The downloaded file is not a valid ZIP archive.")
        return ""


# Use the function
words = get_movie_words("https://dl.subdl.com/subtitle/387706-1955487")

# If words were extracted, print the word count
if words:
    print("Word count:", len(words.split()))
