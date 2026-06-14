# -*- coding: utf-8 -*-

class Config(object):
    def __init__(self, root_path='.', test_csv='Protocols/PolyU/xspectral_test.csv',
                 num_class=209, data_name='PolyU_XSpectral'):
        self._root_path = root_path
        self._test_list = test_csv
        self._num_class = num_class
        self.data_name = data_name
        self.test_type = 'CrossSpectral'

    def num_classGet(self):
        return self._num_class

    def load_detailGet(self):
        return self._root_path, self._test_list

    def test_loaderGet(self):
        return self._root_path, self._test_list
