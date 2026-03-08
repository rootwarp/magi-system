"""Content extraction tool for fetching and extracting text from web pages."""

import trafilatura


def extract_page_content(url: str) -> dict:
    """Extract the main text content from a web page URL.

    Use this tool when you need to read and understand the content of a web page.
    It fetches the page and extracts the main text content, filtering out
    navigation, ads, and other boilerplate. Links and tables are preserved.

    Args:
        url: The URL of the web page to extract content from.

    Returns:
        A dictionary with 'url' and 'content' keys on success,
        or 'error', 'url', and empty 'content' on failure.
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return {
                "error": f"Failed to fetch URL: {url}",
                "url": url,
                "content": "",
            }

        content = trafilatura.extract(
            downloaded,
            include_links=True,
            include_tables=True,
            favor_precision=True,
        )

        if not content:
            return {
                "error": f"No content could be extracted from: {url}",
                "url": url,
                "content": "",
            }

        return {"url": url, "content": content}
    except Exception as e:
        return {
            "error": str(e),
            "url": url,
            "content": "",
        }
