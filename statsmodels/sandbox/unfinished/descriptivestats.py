
if __name__ == "__main__":
    #unittest.main()

    data4 = np.array([[1,2,3,4,5,6],
                      [6,5,4,3,2,1],
                      [9,9,9,9,9,9]])

    t1 = Describe(data4)
    #print(t1.summary(stats='all'))
    noperc = ['obs', 'mean', 'std', 'min', 'max', 'ptp', #'mode',  #'var',
                        'median', 'skew', 'uss', 'kurtosis']
    #TODO: mode var raise exception,
    #TODO: percentile writes list in cell (?), huge wide format
    print(t1.summary(stats=noperc))
    print(t1.summary())
    print(t1.summary( orientation='varcols'))
    print(t1.summary(stats=['mean', 'median', 'min', 'max'], orientation=('varcols')))
    print(t1.summary(stats='all'))


    data1 = np.array([(1,2,'a','aa'),
                      (2,3,'b','bb'),
                      (2,4,'b','cc')],
                     dtype = [('alpha',float), ('beta', int),
                              ('gamma', '|S1'), ('delta', '|S2')])
    data2 = np.array([(1,2),
                      (2,3),
                      (2,4)],
                     dtype = [('alpha',float), ('beta', float)])

    data3 = np.array([[1,2,4,4],
                      [2,3,3,3],
                      [2,4,4,3]], dtype=float)


    class TestSimpleTable(object):
        #from statsmodels.iolib.table import SimpleTable, default_txt_fmt

        def test_basic_1(self):
            print('test_basic_1')
            t1 = Describe(data1)
            print(t1.summary())


        def test_basic_2(self):
            print('test_basic_2')
            t2 = Describe(data2)
            print(t2.summary())

        def test_basic_3(self):
            print('test_basic_3')
            t1 = Describe(data3)
            print(t1.summary())

        def test_basic_4(self):
            print('test_basic_4')
            t1 = Describe(data4)
            print(t1.summary())

        def test_basic_1a(self):
            print('test_basic_1a')
            t1 = Describe(data1)
            print(t1.summary(stats='basic', columns=['alpha']))

        def test_basic_1b(self):
            print('test_basic_1b')
            t1 = Describe(data1)
            print(t1.summary(stats='basic', columns='all'))

        def test_basic_2a(self):
            print('test_basic_2a')
            t2 = Describe(data2)
            print(t2.summary(stats='all'))

        def test_basic_3(aself):
            t1 = Describe(data3)
            print(t1.summary(stats='all'))

        def test_basic_4a(self):
            t1 = Describe(data4)
            print(t1.summary(stats='all'))


