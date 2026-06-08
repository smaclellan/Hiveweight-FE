import pytest
import hiveweight


@pytest.fixture(autouse=True)
def clear_etag_cache():
    """Reset the module-level ETag cache before and after every test."""
    hiveweight.ETAG_CACHE.clear()
    yield
    hiveweight.ETAG_CACHE.clear()
