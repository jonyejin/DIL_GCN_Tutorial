import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        from numpy import load
        data = load('Data Files/cora_split_0.6_0.2_0.npz')
        lst = data.files

        for item in lst:
            print(item)
            print(sum(data[item]))
            print(data[item])



if __name__ == '__main__':
    unittest.main()
