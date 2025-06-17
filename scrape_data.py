import os
import requests
import json
from datetime import datetime
import sqlite3
from dotenv import load_dotenv
import numpy as np
import time

load_dotenv()

DISCOURSE_URL = "https://discourse.onlinedegree.iitm.ac.in"
GITHUB_API = "https://api.github.com/repos/sanand0/tools-in-data-science-public/contents"
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
DISCOURSE_COOKIES = {
    "_t": os.environ.get("DISCOURSE_T_COOKIE"),
    "_forum_session": os.environ.get("DISCOURSE_SESSION_COOKIE")
}

def get_embedding(text):
    """Fetch embedding for text using AI Proxy."""
    url = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    payload = {
        "model": "text-embedding-3-small",
        "input": text[:8192]  # Truncate to avoid token limits
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    except Exception as e:
        print(f"Embedding error for text starting with: {text[:50]}...: {e}")
        return None

def scrape_github():
    """Scrape markdown and text files from GitHub repo."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"Bearer {GITHUB_TOKEN}" if GITHUB_TOKEN else None
    }
    for attempt in range(3):
        try:
            response = requests.get(GITHUB_API, headers=headers, timeout=10)
            response.raise_for_status()
            files = response.json()
            content = []
            for file in files:
                if file["type"] == "file" and file["name"].endswith((".md", ".txt")):
                    file_response = requests.get(file["download_url"], timeout=10)
                    if file_response.status_code == 200:
                        content.append({
                            "source": "github",
                            "content": file_response.text,
                            "url": file["html_url"],
                            "title": file["name"]
                        })
            print(f"Scraped {len(content)} files from GitHub")
            return content
        except requests.exceptions.HTTPError as e:
            if response.status_code == 503:
                print(f"GitHub 503 error, retrying ({attempt + 1}/3)...")
                time.sleep(2 ** attempt)
                continue
            print(f"GitHub scraping error: {e}")
            print(f"GitHub response: {response.text}")
            return []
        except Exception as e:
            print(f"GitHub scraping error: {e}")
            return []
    print("GitHub scraping failed after 3 attempts")
    return []

def scrape_discourse(start_date, end_date):
    """Scrape Discourse posts from the TDS category within date range."""
    session = requests.Session()
    for name, value in DISCOURSE_COOKIES.items():
        session.cookies.set(name, value, domain=DISCOURSE_URL.split("//")[1])
    
    try:
        # Verify authentication
        response = session.get(f"{DISCOURSE_URL}/session/current.json", timeout=10)
        response.raise_for_status()
        print(f"Authenticated as: {response.json()['current_user']['username']}")
        
        # Find TDS category
        response = session.get(f"{DISCOURSE_URL}/categories.json", timeout=10)
        response.raise_for_status()
        categories = response.json()["category_list"]["categories"]
        category_id = None
        for cat in categories:
            print(f"Category: {cat['name']}, ID: {cat['id']}")
            if any(keyword in cat["name"].lower() for keyword in ["tools in data science", "tds", "data science"]):
                category_id = cat["id"]
                print(f"Found TDS category: {cat['name']} (ID: {category_id})")
                break
        if not category_id:
            category_id = 9  # Try 'Courses' category as parent
            print(f"Using fallback category ID: {category_id} (Courses). Checking subcategories...")
            response = session.get(f"{DISCOURSE_URL}/c/{category_id}.json", timeout=10)
            response.raise_for_status()
            subcategories = response.json().get("topic_list", {}).get("subcategories", [])
            for subcat in subcategories:
                print(f"Subcategory: {subcat['name']}, Slug: {subcat['slug']}")
                if "tools-in-data-science" in subcat["slug"].lower():
                    category_id = subcat["id"]
                    print(f"Found TDS subcategory: {subcat['name']} (ID: {category_id})")
                    break
        if not category_id:
            print("TDS category not found. Skipping Discourse scraping. Please verify the category ID manually.")
            return []
        
        # Fetch topics
        posts = []
        response = session.get(f"{DISCOURSE_URL}/c/{category_id}.json", timeout=10)
        response.raise_for_status()
        topics = response.json().get("topic_list", {}).get("topics", [])
        
        for topic in topics:
            created_at = datetime.strptime(topic["created_at"].split("T")[0], "%Y-%m-%d")
            if start_date <= created_at <= end_date:
                topic_id = topic["id"]
                topic_title = topic["title"]
                response = session.get(f"{DISCOURSE_URL}/t/{topic_id}.json", timeout=10)
                if response.status_code == 200:
                    topic_data = response.json()
                    for post in topic_data.get("post_stream", {}).get("posts", []):
                        post_date = datetime.strptime(post["created_at"], "%Y-%m-%dT%H:%M:%S.%fZ")
                        if start_date <= post_date <= end_date:
                            posts.append({
                                "source": "discourse",
                                "content": post["cooked"],
                                "url": f"{DISCOURSE_URL}/t/{topic_id}/{post['post_number']}",
                                "title": topic_title
                            })
        print(f"Scraped {len(posts)} posts from Discourse")
        return posts
    except Exception as e:
        print(f"Discourse scraping error: {e}")
        return []

def main():
    """Main function to scrape data and store in SQLite."""
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 6, 18)  # Test range for current date
    
    # Initialize database
    conn = sqlite3.connect("tds_data.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS content 
                 (id INTEGER PRIMARY KEY, source TEXT, content TEXT, url TEXT, title TEXT, embedding TEXT)''')
    conn.commit()
    
    # Scrape data
    github_content = scrape_github()
    discourse_posts = scrape_discourse(start_date, end_date)
    all_content = github_content + discourse_posts
    
    # Compute and store embeddings
    stored_count = 0
    for item in all_content:
        embedding = get_embedding(item["content"])
        # Store even if embedding fails
        c.execute("INSERT INTO content (source, content, url, title, embedding) VALUES (?, ?, ?, ?, ?)",
                  (item["source"], item["content"], item["url"], item["title"], json.dumps(embedding) if embedding else None))
        stored_count += 1
    
    conn.commit()
    conn.close()
    print(f"Stored {stored_count} items in tds_data.db")

if __name__ == "__main__":
    main()