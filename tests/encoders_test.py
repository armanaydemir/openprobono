"""Tests for different encoder models."""
import unittest

from app import encoders
from app.models import EncoderParams


class EncoderTests(unittest.TestCase):
    """Test class for different encoder models."""

    # 1024 dimensions, 512 max tokens
    test_model = EncoderParams(name="WhereIsAI/UAE-Large-V1", dim=1024)
