import argparse
import sys
import subprocess
import re

def load_index(filepath="llms.txt"):
    """Parses llms.txt to get a map of Title -> URL"""
    index = {}
    with open(filepath, 'r') as f:
        content = f.read()

    # Simple regex to find [Title](URL)
    matches = re.findall(r'\[(.*?)\]\((.*?)\)', content)
    for title, url in matches:
        index[title] = url
    return index

def fetch_content(url):
    """Fetches content using curl"""
    print(f"Fetching: {url}...")
    try:
        # Using curl -L to follow redirects
        result = subprocess.run(['curl', '-L', '-s', url], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Error fetching {url}: {result.stderr}"
    except Exception as e:
        return f"Exception: {e}"

def search_index(query, index):
    """Searches keys in the index"""
    results = {}
    for title, url in index.items():
        if query.lower() in title.lower():
            results[title] = url
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangChain Skills Fetcher")
    subparsers = parser.add_subparsers(dest="command")

    # List command
    list_parser = subparsers.add_parser("list", help="List all available docs")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search docs by keyword")
    search_parser.add_argument("query", help="Keyword to search for")

    # Fetch command
    fetch_parser = subparsers.add_parser("fetch", help="Fetch doc content by exact title or URL")
    fetch_parser.add_argument("target", help="Title or URL to fetch")

    args = parser.parse_args()

    index = load_index()

    if args.command == "list":
        for title, url in index.items():
            print(f"{title}: {url}")

    elif args.command == "search":
        results = search_index(args.query, index)
        if results:
            print(f"Found {len(results)} matches:")
            for title, url in results.items():
                print(f"  {title}: {url}")
        else:
            print("No matches found.")

    elif args.command == "fetch":
        target = args.target
        url = target

        # If target is a title in our index, use the mapped URL
        if target in index:
            url = index[target]
        # If target is part of a title (imprecise fetch), try to find it
        elif target not in index and not target.startswith("http"):
             results = search_index(target, index)
             if len(results) == 1:
                 url = list(results.values())[0]
                 print(f"Resolved '{target}' to {url}")
             elif len(results) > 1:
                 print(f"Ambiguous target '{target}'. Matches: {list(results.keys())}")
                 sys.exit(1)

        content = fetch_content(url)
        print(content)

    else:
        parser.print_help()
