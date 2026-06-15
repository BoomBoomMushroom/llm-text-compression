import requests
import time
import os
from pathlib import Path

userAgentHeaders = {
    "User-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:151.0) Gecko/20100101 Firefox/151.0"
}

# I found this great API (https://wikitext.eluni.co/api/extract?format=text&?url=) at https://wikitext.eluni.co/

wikipediaFolder = "./data/wikipedia"

def getFeaturedWikipediaPage() -> str:
    randomFeaturedPageUrl = "https://randomincategory.toolforge.org/Featured_articles?site=en.wikipedia.org"
    r = requests.get(randomFeaturedPageUrl, headers=userAgentHeaders)
    featuredPageUrl = r.history[0].headers["location"]
    return featuredPageUrl

def getWikipediaPageText(url: str) -> str:
    r = requests.get(f"https://wikitext.eluni.co/api/extract?format=text&url={url}", userAgentHeaders)
    if r.status_code != 200:
        print(r.status_code)
        print(r.text)
        # probably 429, too many requests. max we can make is 5 in 1 min
        return None
    return r.text

def isPageBad(text: str) -> bool:
    if len(text) < 100: return True
    return False

def downloadAndSaveRandomArticle() -> bool:
    pageUrl = getFeaturedWikipediaPage()
    pageText = getWikipediaPageText(pageUrl)
    if pageText == None:
        # uh oh the request failed! probably rate limited. we shall wait
        print("Get page text failed... waiting")
        time.sleep( 60/5 ) # 5 requests per minute
        return False
    
    isBadPage = isPageBad(pageText)

    if isBadPage:
        print("\nBad page!")
        print(pageText, "\n")
        return False

    pageName = pageUrl.split("/")[-1]
    filePath = f"{wikipediaFolder}/{pageName}.txt"
    if os.path.isfile(filePath):
        print(f"Already downloaded: {filePath}")
        return False
    
    with open(filePath, "w") as f:
        f.write(pageText)
        print(f"Successfully downloaded {pageName}")
        return True

def downloadNPages(n: int=0):
    downloaded = 0
    while downloaded < n:
        print(f"({downloaded} / {n}) | ", end="")
        didDownload = downloadAndSaveRandomArticle()
        if didDownload: downloaded += 1

downloadNPages(1000)

