import unittest

import Lenstreenemy


class VersionTestCase(unittest.TestCase):
    """ Version tests """

    def test_version(self):
        """ check Lenstreenemy exposes a version attribute """
        self.assertTrue(hasattr(Lenstreenemy, "__version__"))
        self.assertIsInstance(Lenstreenemy.__version__, str)


if __name__ == "__main__":
    unittest.main()
