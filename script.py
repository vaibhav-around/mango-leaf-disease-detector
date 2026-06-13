from icrawler.builtin import BingImageCrawler
import os

keywords = [
    "healthy mango leaf",
    "healthy mango leaf close up",
"healthy mango leaves plant pathology",
"healthy mango leaf isolated",
"mango leaf healthy disease free"
]

for keyword in keywords:

    folder = os.path.join(
        "evaluation-dataset",
        keyword.replace(" ", "_")
    )

    os.makedirs(folder, exist_ok=True)

    crawler = BingImageCrawler(
        storage={"root_dir": folder}
    )

    print(f"Downloading: {keyword}")

    crawler.crawl(
        keyword=keyword,
        max_num=20
    )

print("Done!")