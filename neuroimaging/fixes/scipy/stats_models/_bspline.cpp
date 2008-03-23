#ifdef __CPLUSPLUS__
extern "C" {
#endif

#ifndef __GNUC__
#pragma warning(disable: 4275)
#pragma warning(disable: 4101)

#endif
#include "Python.h"
#include "compile.h"
#include "frameobject.h"
#include <complex>
#include <math.h>
#include <string>
#include "scxx/object.h"
#include "scxx/list.h"
#include "scxx/tuple.h"
#include "scxx/dict.h"
#include <iostream>
#include <stdio.h>
#include "numpy/arrayobject.h"




// global None value for use in functions.
namespace py {
object None = object(Py_None);
}

char* find_type(PyObject* py_obj)
{
    if(py_obj == NULL) return "C NULL value";
    if(PyCallable_Check(py_obj)) return "callable";
    if(PyString_Check(py_obj)) return "string";
    if(PyInt_Check(py_obj)) return "int";
    if(PyFloat_Check(py_obj)) return "float";
    if(PyDict_Check(py_obj)) return "dict";
    if(PyList_Check(py_obj)) return "list";
    if(PyTuple_Check(py_obj)) return "tuple";
    if(PyFile_Check(py_obj)) return "file";
    if(PyModule_Check(py_obj)) return "module";

    //should probably do more intergation (and thinking) on these.
    if(PyCallable_Check(py_obj) && PyInstance_Check(py_obj)) return "callable";
    if(PyInstance_Check(py_obj)) return "instance";
    if(PyCallable_Check(py_obj)) return "callable";
    return "unkown type";
}

void throw_error(PyObject* exc, const char* msg)
{
 //printf("setting python error: %s\n",msg);
  PyErr_SetString(exc, msg);
  //printf("throwing error\n");
  throw 1;
}

void handle_bad_type(PyObject* py_obj, const char* good_type, const char* var_name)
{
    char msg[500];
    sprintf(msg,"received '%s' type instead of '%s' for variable '%s'",
            find_type(py_obj),good_type,var_name);
    throw_error(PyExc_TypeError,msg);
}

void handle_conversion_error(PyObject* py_obj, const char* good_type, const char* var_name)
{
    char msg[500];
    sprintf(msg,"Conversion Error:, received '%s' type instead of '%s' for variable '%s'",
            find_type(py_obj),good_type,var_name);
    throw_error(PyExc_TypeError,msg);
}


class int_handler
{
public:
    int convert_to_int(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        
        if (!py_obj || !PyInt_Check(py_obj))
            handle_conversion_error(py_obj,"int", name);
        return (int) PyInt_AsLong(py_obj);
    }

    int py_to_int(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyInt_Check(py_obj))
            handle_bad_type(py_obj,"int", name);
        
        return (int) PyInt_AsLong(py_obj);
    }
};

int_handler x__int_handler = int_handler();
#define convert_to_int(py_obj,name) \
        x__int_handler.convert_to_int(py_obj,name)
#define py_to_int(py_obj,name) \
        x__int_handler.py_to_int(py_obj,name)


PyObject* int_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class float_handler
{
public:
    double convert_to_float(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        
        if (!py_obj || !PyFloat_Check(py_obj))
            handle_conversion_error(py_obj,"float", name);
        return PyFloat_AsDouble(py_obj);
    }

    double py_to_float(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyFloat_Check(py_obj))
            handle_bad_type(py_obj,"float", name);
        
        return PyFloat_AsDouble(py_obj);
    }
};

float_handler x__float_handler = float_handler();
#define convert_to_float(py_obj,name) \
        x__float_handler.convert_to_float(py_obj,name)
#define py_to_float(py_obj,name) \
        x__float_handler.py_to_float(py_obj,name)


PyObject* float_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class complex_handler
{
public:
    std::complex<double> convert_to_complex(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        
        if (!py_obj || !PyComplex_Check(py_obj))
            handle_conversion_error(py_obj,"complex", name);
        return std::complex<double>(PyComplex_RealAsDouble(py_obj),PyComplex_ImagAsDouble(py_obj));
    }

    std::complex<double> py_to_complex(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyComplex_Check(py_obj))
            handle_bad_type(py_obj,"complex", name);
        
        return std::complex<double>(PyComplex_RealAsDouble(py_obj),PyComplex_ImagAsDouble(py_obj));
    }
};

complex_handler x__complex_handler = complex_handler();
#define convert_to_complex(py_obj,name) \
        x__complex_handler.convert_to_complex(py_obj,name)
#define py_to_complex(py_obj,name) \
        x__complex_handler.py_to_complex(py_obj,name)


PyObject* complex_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class unicode_handler
{
public:
    Py_UNICODE* convert_to_unicode(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        Py_XINCREF(py_obj);
        if (!py_obj || !PyUnicode_Check(py_obj))
            handle_conversion_error(py_obj,"unicode", name);
        return PyUnicode_AS_UNICODE(py_obj);
    }

    Py_UNICODE* py_to_unicode(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyUnicode_Check(py_obj))
            handle_bad_type(py_obj,"unicode", name);
        Py_XINCREF(py_obj);
        return PyUnicode_AS_UNICODE(py_obj);
    }
};

unicode_handler x__unicode_handler = unicode_handler();
#define convert_to_unicode(py_obj,name) \
        x__unicode_handler.convert_to_unicode(py_obj,name)
#define py_to_unicode(py_obj,name) \
        x__unicode_handler.py_to_unicode(py_obj,name)


PyObject* unicode_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class string_handler
{
public:
    std::string convert_to_string(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        Py_XINCREF(py_obj);
        if (!py_obj || !PyString_Check(py_obj))
            handle_conversion_error(py_obj,"string", name);
        return std::string(PyString_AsString(py_obj));
    }

    std::string py_to_string(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyString_Check(py_obj))
            handle_bad_type(py_obj,"string", name);
        Py_XINCREF(py_obj);
        return std::string(PyString_AsString(py_obj));
    }
};

string_handler x__string_handler = string_handler();
#define convert_to_string(py_obj,name) \
        x__string_handler.convert_to_string(py_obj,name)
#define py_to_string(py_obj,name) \
        x__string_handler.py_to_string(py_obj,name)


               PyObject* string_to_py(std::string s)
               {
                   return PyString_FromString(s.c_str());
               }
               
class list_handler
{
public:
    py::list convert_to_list(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        
        if (!py_obj || !PyList_Check(py_obj))
            handle_conversion_error(py_obj,"list", name);
        return py::list(py_obj);
    }

    py::list py_to_list(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyList_Check(py_obj))
            handle_bad_type(py_obj,"list", name);
        
        return py::list(py_obj);
    }
};

list_handler x__list_handler = list_handler();
#define convert_to_list(py_obj,name) \
        x__list_handler.convert_to_list(py_obj,name)
#define py_to_list(py_obj,name) \
        x__list_handler.py_to_list(py_obj,name)


PyObject* list_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class dict_handler
{
public:
    py::dict convert_to_dict(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        
        if (!py_obj || !PyDict_Check(py_obj))
            handle_conversion_error(py_obj,"dict", name);
        return py::dict(py_obj);
    }

    py::dict py_to_dict(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyDict_Check(py_obj))
            handle_bad_type(py_obj,"dict", name);
        
        return py::dict(py_obj);
    }
};

dict_handler x__dict_handler = dict_handler();
#define convert_to_dict(py_obj,name) \
        x__dict_handler.convert_to_dict(py_obj,name)
#define py_to_dict(py_obj,name) \
        x__dict_handler.py_to_dict(py_obj,name)


PyObject* dict_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class tuple_handler
{
public:
    py::tuple convert_to_tuple(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        
        if (!py_obj || !PyTuple_Check(py_obj))
            handle_conversion_error(py_obj,"tuple", name);
        return py::tuple(py_obj);
    }

    py::tuple py_to_tuple(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyTuple_Check(py_obj))
            handle_bad_type(py_obj,"tuple", name);
        
        return py::tuple(py_obj);
    }
};

tuple_handler x__tuple_handler = tuple_handler();
#define convert_to_tuple(py_obj,name) \
        x__tuple_handler.convert_to_tuple(py_obj,name)
#define py_to_tuple(py_obj,name) \
        x__tuple_handler.py_to_tuple(py_obj,name)


PyObject* tuple_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class file_handler
{
public:
    FILE* convert_to_file(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        Py_XINCREF(py_obj);
        if (!py_obj || !PyFile_Check(py_obj))
            handle_conversion_error(py_obj,"file", name);
        return PyFile_AsFile(py_obj);
    }

    FILE* py_to_file(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyFile_Check(py_obj))
            handle_bad_type(py_obj,"file", name);
        Py_XINCREF(py_obj);
        return PyFile_AsFile(py_obj);
    }
};

file_handler x__file_handler = file_handler();
#define convert_to_file(py_obj,name) \
        x__file_handler.convert_to_file(py_obj,name)
#define py_to_file(py_obj,name) \
        x__file_handler.py_to_file(py_obj,name)


               PyObject* file_to_py(FILE* file, char* name, char* mode)
               {
                   return (PyObject*) PyFile_FromFile(file, name, mode, fclose);
               }
               
class instance_handler
{
public:
    py::object convert_to_instance(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        
        if (!py_obj || !PyInstance_Check(py_obj))
            handle_conversion_error(py_obj,"instance", name);
        return py::object(py_obj);
    }

    py::object py_to_instance(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyInstance_Check(py_obj))
            handle_bad_type(py_obj,"instance", name);
        
        return py::object(py_obj);
    }
};

instance_handler x__instance_handler = instance_handler();
#define convert_to_instance(py_obj,name) \
        x__instance_handler.convert_to_instance(py_obj,name)
#define py_to_instance(py_obj,name) \
        x__instance_handler.py_to_instance(py_obj,name)


PyObject* instance_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class numpy_size_handler
{
public:
    void conversion_numpy_check_size(PyArrayObject* arr_obj, int Ndims,
                                     const char* name)
    {
        if (arr_obj->nd != Ndims)
        {
            char msg[500];
            sprintf(msg,"Conversion Error: received '%d' dimensional array instead of '%d' dimensional array for variable '%s'",
                    arr_obj->nd,Ndims,name);
            throw_error(PyExc_TypeError,msg);
        }
    }

    void numpy_check_size(PyArrayObject* arr_obj, int Ndims, const char* name)
    {
        if (arr_obj->nd != Ndims)
        {
            char msg[500];
            sprintf(msg,"received '%d' dimensional array instead of '%d' dimensional array for variable '%s'",
                    arr_obj->nd,Ndims,name);
            throw_error(PyExc_TypeError,msg);
        }
    }
};

numpy_size_handler x__numpy_size_handler = numpy_size_handler();
#define conversion_numpy_check_size x__numpy_size_handler.conversion_numpy_check_size
#define numpy_check_size x__numpy_size_handler.numpy_check_size


class numpy_type_handler
{
public:
    void conversion_numpy_check_type(PyArrayObject* arr_obj, int numeric_type,
                                     const char* name)
    {
        // Make sure input has correct numeric type.
        int arr_type = arr_obj->descr->type_num;
        if (PyTypeNum_ISEXTENDED(numeric_type))
        {
        char msg[80];
        sprintf(msg, "Conversion Error: extended types not supported for variable '%s'",
                name);
        throw_error(PyExc_TypeError, msg);
        }
        if (!PyArray_EquivTypenums(arr_type, numeric_type))
        {

        char* type_names[23] = {"bool", "byte", "ubyte","short", "ushort",
                                "int", "uint", "long", "ulong", "longlong", "ulonglong",
                                "float", "double", "longdouble", "cfloat", "cdouble",
                                "clongdouble", "object", "string", "unicode", "void", "ntype",
                                "unknown"};
        char msg[500];
        sprintf(msg,"Conversion Error: received '%s' typed array instead of '%s' typed array for variable '%s'",
                type_names[arr_type],type_names[numeric_type],name);
        throw_error(PyExc_TypeError,msg);
        }
    }

    void numpy_check_type(PyArrayObject* arr_obj, int numeric_type, const char* name)
    {
        // Make sure input has correct numeric type.
        int arr_type = arr_obj->descr->type_num;
        if (PyTypeNum_ISEXTENDED(numeric_type))
        {
        char msg[80];
        sprintf(msg, "Conversion Error: extended types not supported for variable '%s'",
                name);
        throw_error(PyExc_TypeError, msg);
        }
        if (!PyArray_EquivTypenums(arr_type, numeric_type))
        {
            char* type_names[23] = {"bool", "byte", "ubyte","short", "ushort",
                                    "int", "uint", "long", "ulong", "longlong", "ulonglong",
                                    "float", "double", "longdouble", "cfloat", "cdouble",
                                    "clongdouble", "object", "string", "unicode", "void", "ntype",
                                    "unknown"};
            char msg[500];
            sprintf(msg,"received '%s' typed array instead of '%s' typed array for variable '%s'",
                    type_names[arr_type],type_names[numeric_type],name);
            throw_error(PyExc_TypeError,msg);
        }
    }
};

numpy_type_handler x__numpy_type_handler = numpy_type_handler();
#define conversion_numpy_check_type x__numpy_type_handler.conversion_numpy_check_type
#define numpy_check_type x__numpy_type_handler.numpy_check_type


class numpy_handler
{
public:
    PyArrayObject* convert_to_numpy(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        Py_XINCREF(py_obj);
        if (!py_obj || !PyArray_Check(py_obj))
            handle_conversion_error(py_obj,"numpy", name);
        return (PyArrayObject*) py_obj;
    }

    PyArrayObject* py_to_numpy(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyArray_Check(py_obj))
            handle_bad_type(py_obj,"numpy", name);
        Py_XINCREF(py_obj);
        return (PyArrayObject*) py_obj;
    }
};

numpy_handler x__numpy_handler = numpy_handler();
#define convert_to_numpy(py_obj,name) \
        x__numpy_handler.convert_to_numpy(py_obj,name)
#define py_to_numpy(py_obj,name) \
        x__numpy_handler.py_to_numpy(py_obj,name)


PyObject* numpy_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class catchall_handler
{
public:
    py::object convert_to_catchall(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        
        if (!py_obj || !(py_obj))
            handle_conversion_error(py_obj,"catchall", name);
        return py::object(py_obj);
    }

    py::object py_to_catchall(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !(py_obj))
            handle_bad_type(py_obj,"catchall", name);
        
        return py::object(py_obj);
    }
};

catchall_handler x__catchall_handler = catchall_handler();
#define convert_to_catchall(py_obj,name) \
        x__catchall_handler.convert_to_catchall(py_obj,name)
#define py_to_catchall(py_obj,name) \
        x__catchall_handler.py_to_catchall(py_obj,name)


PyObject* catchall_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


    double *bspline(double **output, double *x, int nx,
                    double *knots, int nknots,
                    int m, int d, int lower, int upper)
    {
       int nbasis;
       int index, i, j, k;
       double *result, *b, *b0, *b1;
       double *f0, *f1;
       double denom;

       nbasis = upper - lower;

       result = *((double **) output);
       f0 = (double *) malloc(sizeof(*f0) * nx);
       f1 = (double *) malloc(sizeof(*f1) * nx);

       if (m == 1) {
           for(i=0; i<nbasis; i++) {
               index = i + lower;

               if(index < nknots - 1) {
                   if ((knots[index] != knots[index+1]) && (d <= 0)) {
                       for (k=0; k<nx; k++) {

                           *result = (double) (x[k] >= knots[index]) * (x[k] < knots[index+1]);
                           result++;
                       }
                   }
                   else {
                       for (k=0; k<nx; k++) {
                           *result = 0.;
                           result++;
                       }
                   }
                }
                else {
                   for (k=0; k<nx; k++) {
                       *result = 0.;
                       result++;
                   }
               }
            }
        }
        else {
            b = (double *) malloc(sizeof(*b) * (nbasis+1) * nx);
            bspline(&b, x, nx, knots, nknots, m-1, d-1, lower, upper+1);

            for(i=0; i<nbasis; i++) {
                b0 = b + nx*i;
                b1 = b + nx*(i+1);

                index = i+lower;

                if ((knots[index] != knots[index+m-1]) && (index+m-1 < nknots)) {
                    denom = knots[index+m-1] - knots[index];
                    if (d <= 0) {
                        for (k=0; k<nx; k++) {
                            f0[k] = (x[k] - knots[index]) / denom;
                        }
                    }
                    else {
                        for (k=0; k<nx; k++) {
                            f0[k] = (m-1) / (knots[index+m-1] - knots[index]);
                        }
                    }
                }
                else {
                    for (k=0; k<nx; k++) {
                        f0[k] = 0.;
                    }
                }

                index = i+lower+1;
                if ((knots[index] != knots[index+m-1]) && (index+m-1 < nknots)) {
                    denom = knots[index+m-1] - knots[index];
                    if (d <= 0) {
                        for (k=0; k<nx; k++) {
                            f1[k] = (knots[index+m-1] - x[k]) / denom;
                        }
                    }
                    else {
                        for (k=0; k<nx; k++) {
                            f1[k] = -(m-1) / (knots[index+m-1] - knots[index]);
                        }
                    }
                }
                else {
                    for (k=0; k<nx; k++) {
                        f1[k] = 0.;
                    }
                }

                for (k=0; k<nx; k++) {
                    *result = f0[k]*(*b0) + f1[k]*(*b1);
                    b0++; b1++; result++;
                }
            }
            free(b);
        }
        free(f0); free(f1);
        result = result - nx * nbasis;

        return(result);
    }
    

    double *bspline_prod(double *x, int nx, double *knots, int nknots,
                        int m, int l, int r, int dl, int dr)
    {
        double *result, *bl, *br;
        int k;

        if (fabs(r - l) <= m) {
            result = (double *) malloc(sizeof(*result) * nx);
            bl = (double *) malloc(sizeof(*bl) * nx);
            br = (double *) malloc(sizeof(*br) * nx);

            bl = bspline(&bl, x, nx, knots, nknots, m, dl, l, l+1);
            br = bspline(&br, x, nx, knots, nknots, m, dr, r, r+1);

            for (k=0; k<nx; k++) {
                result[k] = bl[k] * br[k];
            }
            free(bl); free(br);
        }
        else {
            for (k=0; k<nx; k++) {
                result[k] = 0.;
            }
        }

        return(result);
    }


    double bspline_quad(double *knots, int nknots,
                        int m, int l, int r, int dl, int dr)

        /* This is based on scipy.integrate.fixed_quad */

    {
        double *y;
        double qx[18]={-0.99156516842093112, -0.95582394957139805, -0.89260246649755559, -0.80370495897252259, -0.69168704306035289, -0.55977083107394709, -0.41175116146284241, -0.25188622569150537, -0.084775013041735167, 0.084775013041735375, 0.25188622569150537, 0.41175116146284252, 0.55977083107394776, 0.69168704306035311, 0.80370495897252336, 0.89260246649755637, 0.95582394957139805, 0.99156516842093201};
        double qw[18]={0.021616013526483319, 0.049714548894970602, 0.076425730254888843, 0.10094204410628758, 0.12255520671147767, 0.14064291467065076, 0.15468467512626574, 0.16427648374583334, 0.16914238296314343, 0.16914238296314382, 0.16427648374583262, 0.15468467512626574, 0.1406429146706504, 0.1225552067114785, 0.10094204410628696, 0.076425730254889288, 0.049714548894969811, 0.021616013526484224};
        double x[18];
        int nq=18;
        int k, kk;
        int lower, upper;
        double result, a, b, partial;

        result = 0;

        /* TO DO: figure out knot span more efficiently */

        lower = l - m - 1;
        if (lower < 0) { lower = 0;}
        upper = lower + 2 * m + 4;
        if (upper > nknots - 1) { upper = nknots-1; }

        for (k=lower; k<upper; k++) {
            partial = 0.;
            a = knots[k]; b=knots[k+1];
            for (kk=0; kk<nq; kk++) {
               x[kk] = (b - a) * (qx[kk] + 1) / 2. + a;
            }

            y = bspline_prod(x, nq, knots, nknots, m, l, r, dl, dr);

            for (kk=0; kk<nq; kk++) {
                partial += y[kk] * qw[kk];
            }
            free(y); /* bspline_prod malloc's memory, but does not free it */

            result += (b - a) * partial / 2.;

        }

        return(result);
    }

    void bspline_gram(double **output, double *knots, int nknots,
                        int m, int dl, int dr)

    /* Presumes that the first m and last m knots are to be ignored, i.e.
    the interior knots are knots[(m+1):-(m+1)] and the boundary knots are
    knots[m] and knots[-m]. In this setting the first basis element of interest
    is the 1st not the 0th. Should maybe be fixed? */

    {
        double *result;
        int l, r, i, j;
        int nbasis;

        nbasis = nknots - m;

        result = *((double **) output);
        for (i=0; i<nbasis; i++) {
            for (j=0; j<m; j++) {
                l = i;
                r = l+j;
                *result = bspline_quad(knots, nknots, m, l, r, dl, dr);
                result++;
            }
        }
    }

    

    void invband_compute(double **dataptr, double *L, int n, int m) {

        /* Note: m is number of bands not including the diagonal so L is of size (m+1)xn */

        int i,j,k;
        int idx, idy;
        double *data, *odata;
        double diag;

        data = *((double **) dataptr);

        for (i=0; i<n; i++) {
             diag = L[i];
             data[i] = 1.0 / (diag*diag) ;

             for (j=0; j<=m; j++) {
                 L[j*n+i] /= diag;
                 if (j > 0) { data[j*n+i] = 0;}
             }
         }

        for (i=n-1; i>=0; i--) {
             for (j=1; j <= (m<n-1-i ? m:n-1-i); j++) {
                  for (k=1; k<=(n-1-i<m ? n-1-i:m); k++) {
                      idx = (j<k ? k-j:j-k); idy = (j<k ? i+j:i+k);
                      data[j*n+i] -= L[k*n+i] * data[idx*n+idy];
                  }
             }

             for (k=1; k<=(n-1-i<m ? n-1-i:m); k++) {
                  data[i] -= L[k*n+i] * data[k*n+i];
             }
        }

    return;
    }
    

static PyObject* evaluate(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occured = 0;
    PyObject *py_local_dict = NULL;
    static char *kwlist[] = {"x","knots","m","d","lower","upper","local_dict", NULL};
    PyObject *py_x, *py_knots, *py_m, *py_d, *py_lower, *py_upper;
    int x_used, knots_used, m_used, d_used, lower_used, upper_used;
    py_x = py_knots = py_m = py_d = py_lower = py_upper = NULL;
    x_used= knots_used= m_used= d_used= lower_used= upper_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:evaluate",kwlist,&py_x, &py_knots, &py_m, &py_d, &py_lower, &py_upper, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_x = py_x;
        PyArrayObject* x_array = convert_to_numpy(py_x,"x");
        conversion_numpy_check_type(x_array,PyArray_DOUBLE,"x");
        #define X1(i) (*((double*)(x_array->data + (i)*Sx[0])))
        #define X2(i,j) (*((double*)(x_array->data + (i)*Sx[0] + (j)*Sx[1])))
        #define X3(i,j,k) (*((double*)(x_array->data + (i)*Sx[0] + (j)*Sx[1] + (k)*Sx[2])))
        #define X4(i,j,k,l) (*((double*)(x_array->data + (i)*Sx[0] + (j)*Sx[1] + (k)*Sx[2] + (l)*Sx[3])))
        npy_intp* Nx = x_array->dimensions;
        npy_intp* Sx = x_array->strides;
        int Dx = x_array->nd;
        double* x = (double*) x_array->data;
        x_used = 1;
        py_knots = py_knots;
        PyArrayObject* knots_array = convert_to_numpy(py_knots,"knots");
        conversion_numpy_check_type(knots_array,PyArray_DOUBLE,"knots");
        #define KNOTS1(i) (*((double*)(knots_array->data + (i)*Sknots[0])))
        #define KNOTS2(i,j) (*((double*)(knots_array->data + (i)*Sknots[0] + (j)*Sknots[1])))
        #define KNOTS3(i,j,k) (*((double*)(knots_array->data + (i)*Sknots[0] + (j)*Sknots[1] + (k)*Sknots[2])))
        #define KNOTS4(i,j,k,l) (*((double*)(knots_array->data + (i)*Sknots[0] + (j)*Sknots[1] + (k)*Sknots[2] + (l)*Sknots[3])))
        npy_intp* Nknots = knots_array->dimensions;
        npy_intp* Sknots = knots_array->strides;
        int Dknots = knots_array->nd;
        double* knots = (double*) knots_array->data;
        knots_used = 1;
        py_m = py_m;
        int m = convert_to_int(py_m,"m");
        m_used = 1;
        py_d = py_d;
        int d = convert_to_int(py_d,"d");
        d_used = 1;
        py_lower = py_lower;
        int lower = convert_to_int(py_lower,"lower");
        lower_used = 1;
        py_upper = py_upper;
        int upper = convert_to_int(py_upper,"upper");
        upper_used = 1;
        /*<function call here>*/     
        
        
            npy_intp dim[2] = {upper-lower, Nx[0]};
            PyArrayObject *basis;
            double *data;
        
            basis = (PyArrayObject *) PyArray_SimpleNew(2, dim, PyArray_DOUBLE);
            data = (double *) basis->data;
            bspline(&data, x, Nx[0], knots, Nknots[0], m, d, lower, upper);
            return_val = (PyObject *) basis;
            Py_DECREF((PyObject *) basis);
        
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occured = 1;       
    }                                
    /*cleanup code*/                     
    if(x_used)
    {
        Py_XDECREF(py_x);
        #undef X1
        #undef X2
        #undef X3
        #undef X4
    }
    if(knots_used)
    {
        Py_XDECREF(py_knots);
        #undef KNOTS1
        #undef KNOTS2
        #undef KNOTS3
        #undef KNOTS4
    }
    if(!(PyObject*)return_val && !exception_occured)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* gram(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occured = 0;
    PyObject *py_local_dict = NULL;
    static char *kwlist[] = {"knots","m","dl","dr","local_dict", NULL};
    PyObject *py_knots, *py_m, *py_dl, *py_dr;
    int knots_used, m_used, dl_used, dr_used;
    py_knots = py_m = py_dl = py_dr = NULL;
    knots_used= m_used= dl_used= dr_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOO|O:gram",kwlist,&py_knots, &py_m, &py_dl, &py_dr, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_knots = py_knots;
        PyArrayObject* knots_array = convert_to_numpy(py_knots,"knots");
        conversion_numpy_check_type(knots_array,PyArray_DOUBLE,"knots");
        #define KNOTS1(i) (*((double*)(knots_array->data + (i)*Sknots[0])))
        #define KNOTS2(i,j) (*((double*)(knots_array->data + (i)*Sknots[0] + (j)*Sknots[1])))
        #define KNOTS3(i,j,k) (*((double*)(knots_array->data + (i)*Sknots[0] + (j)*Sknots[1] + (k)*Sknots[2])))
        #define KNOTS4(i,j,k,l) (*((double*)(knots_array->data + (i)*Sknots[0] + (j)*Sknots[1] + (k)*Sknots[2] + (l)*Sknots[3])))
        npy_intp* Nknots = knots_array->dimensions;
        npy_intp* Sknots = knots_array->strides;
        int Dknots = knots_array->nd;
        double* knots = (double*) knots_array->data;
        knots_used = 1;
        py_m = py_m;
        int m = convert_to_int(py_m,"m");
        m_used = 1;
        py_dl = py_dl;
        int dl = convert_to_int(py_dl,"dl");
        dl_used = 1;
        py_dr = py_dr;
        int dr = convert_to_int(py_dr,"dr");
        dr_used = 1;
        /*<function call here>*/     
        
        
            npy_intp dim[2] = {Nknots[0]-m, m};
            double *data;
            PyArrayObject *gram;
        
            gram = (PyArrayObject *) PyArray_SimpleNew(2, dim, PyArray_DOUBLE);
            data = (double *) gram->data;
            bspline_gram(&data, knots, Nknots[0], m, dl, dr);
            return_val = (PyObject *) gram;
            Py_DECREF((PyObject *) gram);
        
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occured = 1;       
    }                                
    /*cleanup code*/                     
    if(knots_used)
    {
        Py_XDECREF(py_knots);
        #undef KNOTS1
        #undef KNOTS2
        #undef KNOTS3
        #undef KNOTS4
    }
    if(!(PyObject*)return_val && !exception_occured)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* invband(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occured = 0;
    PyObject *py_local_dict = NULL;
    static char *kwlist[] = {"L","local_dict", NULL};
    PyObject *py_L;
    int L_used;
    py_L = NULL;
    L_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"O|O:invband",kwlist,&py_L, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_L = py_L;
        PyArrayObject* L_array = convert_to_numpy(py_L,"L");
        conversion_numpy_check_type(L_array,PyArray_DOUBLE,"L");
        #define L1(i) (*((double*)(L_array->data + (i)*SL[0])))
        #define L2(i,j) (*((double*)(L_array->data + (i)*SL[0] + (j)*SL[1])))
        #define L3(i,j,k) (*((double*)(L_array->data + (i)*SL[0] + (j)*SL[1] + (k)*SL[2])))
        #define L4(i,j,k,l) (*((double*)(L_array->data + (i)*SL[0] + (j)*SL[1] + (k)*SL[2] + (l)*SL[3])))
        npy_intp* NL = L_array->dimensions;
        npy_intp* SL = L_array->strides;
        int DL = L_array->nd;
        double* L = (double*) L_array->data;
        L_used = 1;
        /*<function call here>*/     
        
        
            npy_intp dim[2] = {NL[0], NL[1]};
            int i, j;
            double *data;
            PyArrayObject *invband;
        
            invband = (PyArrayObject *) PyArray_SimpleNew(2, dim, PyArray_DOUBLE);
            data = (double *) invband->data;
            invband_compute(&data, L, NL[1], NL[0]-1);
        
            return_val = (PyObject *) invband;
            Py_DECREF((PyObject *) invband);
        
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occured = 1;       
    }                                
    /*cleanup code*/                     
    if(L_used)
    {
        Py_XDECREF(py_L);
        #undef L1
        #undef L2
        #undef L3
        #undef L4
    }
    if(!(PyObject*)return_val && !exception_occured)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                


static PyMethodDef compiled_methods[] = 
{
    {"evaluate",(PyCFunction)evaluate , METH_VARARGS|METH_KEYWORDS},
    {"gram",(PyCFunction)gram , METH_VARARGS|METH_KEYWORDS},
    {"invband",(PyCFunction)invband , METH_VARARGS|METH_KEYWORDS},
    {NULL,      NULL}        /* Sentinel */
};

PyMODINIT_FUNC init_bspline(void)
{
    
    Py_Initialize();
    import_array();
    PyImport_ImportModule("numpy");
    (void) Py_InitModule("_bspline", compiled_methods);
}

#ifdef __CPLUSCPLUS__
}
#endif
