import unittest
from profiles import PROFILES, choose_profile


class ProfilesTest(unittest.TestCase):
    def test_size_and_batch_dispatch(self):
        for n in (1,4,8,16):
            self.assertEqual(choose_profile((n,1,128,128),PROFILES),0)
        for n in (1,2,4,8):
            self.assertEqual(choose_profile((n,1,256,256),PROFILES),1)
        for size in (32,512):
            self.assertEqual(choose_profile((1,1,size,size),PROFILES),2)

    def test_invalid_batches_and_channels(self):
        for shape in ((17,1,128,128),(9,1,256,256),(2,1,512,512),(1,3,128,128),(0,1,128,128)):
            with self.assertRaises(ValueError):
                choose_profile(shape,PROFILES)


if __name__ == '__main__':
    unittest.main()
