"""Tests for content extraction tool."""

from unittest.mock import patch

from magi_system.tools.content_extraction import extract_page_content


class TestExtractPageContent:
    """Tests for extract_page_content function."""

    @patch("magi_system.tools.content_extraction.trafilatura")
    def test_successful_extraction(self, mock_trafilatura):
        """Test successful content extraction returns url and content."""
        mock_trafilatura.fetch_url.return_value = "<html><body>Hello</body></html>"
        mock_trafilatura.extract.return_value = "Extracted content"

        result = extract_page_content("https://example.com")

        assert result == {
            "url": "https://example.com",
            "content": "Extracted content",
        }
        mock_trafilatura.fetch_url.assert_called_once_with("https://example.com")
        mock_trafilatura.extract.assert_called_once_with(
            "<html><body>Hello</body></html>",
            include_links=True,
            include_tables=True,
            favor_precision=True,
        )

    @patch("magi_system.tools.content_extraction.trafilatura")
    def test_fetch_url_returns_none(self, mock_trafilatura):
        """Test graceful handling when fetch_url returns None."""
        mock_trafilatura.fetch_url.return_value = None

        result = extract_page_content("https://bad-url.com")

        assert result == {
            "error": "Failed to fetch URL: https://bad-url.com",
            "url": "https://bad-url.com",
            "content": "",
        }
        mock_trafilatura.extract.assert_not_called()

    @patch("magi_system.tools.content_extraction.trafilatura")
    def test_extract_returns_none(self, mock_trafilatura):
        """Test graceful handling when extract returns None."""
        mock_trafilatura.fetch_url.return_value = "<html></html>"
        mock_trafilatura.extract.return_value = None

        result = extract_page_content("https://empty.com")

        assert result == {
            "error": "No content could be extracted from: https://empty.com",
            "url": "https://empty.com",
            "content": "",
        }

    @patch("magi_system.tools.content_extraction.trafilatura")
    def test_extract_returns_empty_string(self, mock_trafilatura):
        """Test graceful handling when extract returns empty string."""
        mock_trafilatura.fetch_url.return_value = "<html></html>"
        mock_trafilatura.extract.return_value = ""

        result = extract_page_content("https://empty.com")

        assert result == {
            "error": "No content could be extracted from: https://empty.com",
            "url": "https://empty.com",
            "content": "",
        }

    @patch("magi_system.tools.content_extraction.trafilatura")
    def test_exception_handling(self, mock_trafilatura):
        """Test graceful handling of unexpected exceptions."""
        mock_trafilatura.fetch_url.side_effect = Exception("Network error")

        result = extract_page_content("https://error.com")

        assert result == {
            "error": "Network error",
            "url": "https://error.com",
            "content": "",
        }

    def test_return_type_is_dict(self):
        """Test that the function has proper type hints."""
        import inspect

        sig = inspect.signature(extract_page_content)
        assert sig.return_annotation is dict
        assert "url" in sig.parameters

    def test_has_docstring(self):
        """Test that the function has a docstring."""
        assert extract_page_content.__doc__ is not None
        assert len(extract_page_content.__doc__) > 0
