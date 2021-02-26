from unittest import TestCase

from code2seq.utils.converting import parse_token


class Test(TestCase):
    def test_parse_token_split(self):
        result = parse_token("abc|123|zxc", True, "|")
        self.assertListEqual(["abc", "123", "zxc"], result)

    def test_parse_token_keep(self):
        result = parse_token("abc|123|zxc", False)
        self.assertListEqual(["abc|123|zxc"], result)
