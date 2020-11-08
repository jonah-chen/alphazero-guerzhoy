#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>
#define SIZE 8

static PyObject* 
is_win(PyObject* self, PyObject* args) {
    PyObject *list3_obj;
    if (!PyArg_ParseTuple(args, "O", &list3_obj))
    return NULL;

    double ***board;

    //Create C arrays from numpy objects:
    int typenum = NPY_DOUBLE;
    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(typenum);
    npy_intp dims[3];
    if (PyArray_AsCArray(&list3_obj, (void ***)&board, dims, 3, descr) < 0) {
    PyErr_SetString(PyExc_TypeError, "error converting to c array");
    return NULL;
    }
    
    int y = SIZE;
    unsigned char draw = 1;
    while (y--)
    {
        int x = SIZE;
        while (x--)
        {
            if (draw && (board[y][x][0] || board[y][x][1]))
                draw = 0;
            // horizontal case:
            if (x + 4 < SIZE)
            {
                if (board[y][x][0] &&
                    board[y][x + 1][0] && 
                    board[y][x + 2][0] &&
                    board[y][x + 3][0] &&
                    board[y][x + 4][0])
                    return Py_BuildValue("i", 1);
                if (board[y][x][1] &&
                    board[y][x + 1][1] && 
                    board[y][x + 2][1] &&
                    board[y][x + 3][1] &&
                    board[y][x + 4][1])
                    return Py_BuildValue("i", 2);
                if (y + 4 < SIZE)
                {
                    if (board[y][x][0] &&
                        board[y + 1][x + 1][0] && 
                        board[y + 2][x + 2][0] &&
                        board[y + 3][x + 3][0] &&
                        board[y + 4][x + 4][0])
                        return Py_BuildValue("i", 1);
                    if (board[y][x][1] &&
                        board[y + 1][x + 1][1] && 
                        board[y + 2][x + 2][1] &&
                        board[y + 3][x + 3][1] &&
                        board[y + 4][x + 4][1])
                        return Py_BuildValue("i", 2);
                }
                if (y - 4 >= 0)
                {
                    if (board[y][x][0] &&
                        board[y - 1][x + 1][0] && 
                        board[y - 2][x + 2][0] &&
                        board[y - 3][x + 3][0] &&
                        board[y - 4][x + 4][0])
                        return Py_BuildValue("i", 1);
                    if (board[y][x][1] &&
                        board[y - 1][x + 1][1] && 
                        board[y - 2][x + 2][1] &&
                        board[y - 3][x + 3][1] &&
                        board[y - 4][x + 4][1])
                        return Py_BuildValue("i", 2);
                }
            }
            // vertical case:
            if (y + 4 < SIZE)
            {
                if (board[y][x][0] &&
                board[y + 1][x][0] &&
                board[y + 2][x][0] &&
                board[y + 3][x][0] &&
                board[y + 4][x][0])
                    return Py_BuildValue("i", 1);
                if (board[y][x][1] &&
                board[y + 1][x][1] &&
                board[y + 2][x][1] &&
                board[y + 3][x][1] &&
                board[y + 4][x][1])
                    return Py_BuildValue("i", 2);
            }
        }
    }
    if (draw)
        return Py_BuildValue("i", 3);
    return Py_BuildValue("i", 0);
}
static PyMethodDef myMethods[] = {
    {"is_win", is_win, METH_VARARGS, "Input numpy array with shape=(SIZE,SIZE,2,) and return the index corresponding to the game state in this array [\"Continue Playing\", \"Black won\", \"White won\", \"Draw\"]"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef myModule = {
    PyModuleDef_HEAD_INIT, 
    "nptrain", 
    "Training module for Gomoku using numpy arrays",
    -1,
    myMethods
};

PyMODINIT_FUNC
PyInit_nptrain(void)
{
    import_array();
    return PyModule_Create(&myModule);
}
