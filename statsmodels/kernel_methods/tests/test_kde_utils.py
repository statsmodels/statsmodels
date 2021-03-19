import pytest
from .. import kde_utils
import numpy as np
import numpy.testing as npt


def test_atleast_2df_fromscalar():
    arr = kde_utils.atleast_2df(2)
    assert arr.shape == (1, 1)
    assert arr[0, 0] == 2


def test_atleast_2df_fromlist():
    lst = [1, 2, 3, 4]
    arr = kde_utils.atleast_2df(lst)
    assert arr.shape == (len(lst), 1)


def test_atleast_2df_multiple_lst():
    lst1 = [1, 2, 3]
    lst2 = [[1, 2], [3, 4], [5, 6], [7, 8]]
    arr1, arr2 = kde_utils.atleast_2df(lst1, lst2)
    assert arr1.shape == (len(lst1), 1)
    assert arr2.shape == (len(lst2), 2)


def test_numpy_trans_all_args():

    @kde_utils.numpy_trans(2, 3, out_dtype=float, in_dtype=int)
    def fct(arr, out):
        out[:, :2] = arr
        out[:, 2] = arr.mean(axis=1)

    x = [[1, 1], [2, 3], [4, 5]]
    res = fct(x)

    assert res.shape == (len(x), 3)


def test_numpy_trans_defaults():

    @kde_utils.numpy_trans(1, 1)
    def fct(arr, out):
        out[:] = arr[:, 0]

    x = [1, 2, 3]
    assert fct(x).shape == (len(x),)


def test_numpy_trans_dim_first():

    @kde_utils.numpy_trans(3, 2)
    def fct(arr, out):
        assert arr.shape[-1] == 3
        assert out.shape[-1] == 2
        out[:] = arr[:, :2]

    x = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [1, 3, 5, 7]])
    assert fct(x).shape == (2, x.shape[1])


def test_numpy_trans_bad_dimension():

    @kde_utils.numpy_trans(3, 2)
    def fct(arr, out):
        out[:] = arr[:, :2]

    with pytest.raises(ValueError):
        fct([1, 2, 3, 4])


def test_numpy_trans_too_many_axis():

    @kde_utils.numpy_trans(3, 2)
    def fct(arr, out):
        out[:] = arr[:, :2]

    with pytest.raises(ValueError):
        fct([[[1]], [[2]], [[3]]])


def test_numpy_trans_invalid_dimensions():
    with pytest.raises(ValueError):

        @kde_utils.numpy_trans(1, -3)
        def fct(arr, out):
            pass


def test_numpy_trans_deduce_input_dimensions():

    @kde_utils.numpy_trans(0, 1)
    def fct(arr, out):
        out[:] = arr.mean(axis=1)

    arr = np.c_[np.r_[0:5:1], np.r_[1:6:1], np.r_[2:7:1]].T
    out = fct(arr)

    assert out.shape == (3,)


def test_numpy_trans1d_all_args():

    @kde_utils.numpy_trans1d(float, int)
    def fct(arr, out):
        out[:] = arr + 0.5

    lst = [1, 2, 3]
    assert fct(lst).shape == (len(lst),)


def test_numpy_trans_method_int_args():

    class Foo(object):
        @kde_utils.numpy_trans_method(2, 2, out_dtype=float, in_dtype=int)
        def fct(self, arr, out):
            assert arr.dtype == np.dtype(int)
            out[:] = arr + 0.5

    f = Foo()
    lst = np.array([[1, 2], [3, 4], [5, 6]])
    out = f.fct(lst)
    assert out.shape == lst.shape
    assert out.dtype == np.dtype(float)


def test_numpy_trans_method_str_args():

    class Foo(object):

        idim = 2
        odim = 3

        @kde_utils.numpy_trans_method('idim', 'odim')
        def fct(self, arr, out):
            assert arr.dtype == np.dtype(float)
            out[:, :self.idim] = arr + 0.5

    f = Foo()
    lst = np.array([[1, 2], [3, 4], [5, 6]])
    out = f.fct(lst)
    assert out.shape == (len(lst), Foo.odim)


def test_numpy_trans_method_bad_args():
    with pytest.raises(ValueError):

        class Foo(object):
            @kde_utils.numpy_trans_method(0, -2)
            def fct(self, arr, out):
                pass


def test_numpy_trans_method_bad_args2():
    with pytest.raises(ValueError):

        class Foo(object):
            @kde_utils.numpy_trans_method(0, -2)
            def fct(self, arr, out):
                pass


def test_axes_tyoe():
    at = kde_utils.AxesType('CUCUO')
    assert str(at) == 'CUCUO'
    assert at[0] == 'C'
    assert at[1:4] == 'UCU'
    assert len(at) == 5
    assert list(at) == list('CUCUO')
    at[0] = 'O'
    assert at[0] == 'O'
    at[:2] = 'CC'
    assert at[:2] == 'CC'
    at1 = at.copy()
    assert all(at1 == at)
    at1[0] = 'A'
    assert any(at1 != at)
    at1.resize(3)
    assert str(at1) == 'ACC'
    at1.resize(5, 'B')
    assert str(at1) == 'ACCBB'


def test_approx_jacobian():

    def func(x, a0, a1, a2):
        return a0 + a1*x + a2*x*x

    J = kde_utils.approx_jacobian([1, 1], func, 1e-6, 1, 2, 3)
    J0 = np.array([[8, 0], [0, 8]])
    npt.assert_allclose(J, J0, rtol=1e-4, atol=1e-4)
