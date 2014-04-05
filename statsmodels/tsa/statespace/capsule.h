/* partially cribbed from numpy/core/include/numpy/npy_3kcompat.h
*/

#include <Python.h>

#if PY_VERSION_HEX >= 0x03000000

void* Capsule_AsVoidPtr(PyObject *obj)
{
    void *ret = PyCapsule_GetPointer(obj, NULL);
    if (ret == NULL) {
        PyErr_Clear();
    }
    return ret;
}

#else

void* Capsule_AsVoidPtr(PyObject *ptr)
{
    return PyCObject_AsVoidPtr(ptr);
}

#endif
