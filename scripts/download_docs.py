import urllib.request
import os

AWS_DOCS = [
    {
        "name": "amazon_bedrock_user_guide",
        "url": "https://docs.aws.amazon.com/pdfs/bedrock/latest/userguide/bedrock-ug.pdf",
        "description": "Amazon Bedrock User Guide"
    },
    {
        "name": "amazon_s3_user_guide",
        "url": "https://docs.aws.amazon.com/pdfs/AmazonS3/latest/userguide/s3-userguide.pdf",
        "description": "Amazon S3 User Guide"
    },
]

DATA_DIR = "data"

def download_doc(name, url, description):
    filepath = os.path.join(DATA_DIR, f"{name}.pdf")
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  ✓ Already exists: {description} ({size_mb:.1f} MB) — skipping")
        return filepath
    print(f"  ↓ Downloading: {description}")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as response:
            with open(filepath, "wb") as f:
                f.write(response.read())
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"    ✓ Done ({size_mb:.1f} MB)")
        return filepath
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        print(f"    → 請手動下載: {url}")
        return None

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("=== Downloading AWS Documentation PDFs ===\n")
    downloaded = []
    for doc in AWS_DOCS:
        path = download_doc(doc["name"], doc["url"], doc["description"])
        if path:
            downloaded.append(path)
    print(f"\n=== 完成: {len(downloaded)}/{len(AWS_DOCS)} 個檔案在 ./{DATA_DIR}/ ===")
    print("\n下一步: python scripts/ingest.py")

if __name__ == "__main__":
    main()