/* C code that contains methods that are used for computations like searching the game board  
 *
 * Author(s): Jonah Chen, Muhammad Ahsan Kaleem
 */

#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>
#include <math.h>
#define SIZE 15

int is_draw(double ***board)
{
    int x, y;
    // Check if there is a win
    y = SIZE;
    while (y--)
    {
        x = SIZE;
        while (x--)
        {
            // horizontal case:
            if (x + 4 < SIZE)
            {
                if (!board[y][x][0] &&
                    !board[y][x + 1][0] && 
                    !board[y][x + 2][0] &&
                    !board[y][x + 3][0] &&
                    !board[y][x + 4][0])
                    return 0;

                if (!board[y][x][1] &&
                    !board[y][x + 1][1] && 
                    !board[y][x + 2][1] &&
                    !board[y][x + 3][1] &&
                    !board[y][x + 4][1])
                    return 0;
                // diagonal case
                if (y + 4 < SIZE)
                {
                    if (!board[y][x][0] &&
                        !board[y + 1][x + 1][0] && 
                        !board[y + 2][x + 2][0] &&
                        !board[y + 3][x + 3][0] &&
                        !board[y + 4][x + 4][0])
                        return 0;
                    if (!board[y][x][1] &&
                        !board[y + 1][x + 1][1] && 
                        !board[y + 2][x + 2][1] &&
                        !board[y + 3][x + 3][1] &&
                        !board[y + 4][x + 4][1])
                        return 0;
                }
                if (y - 4 >= 0)
                {
                    if (!board[y][x][0] &&
                        !board[y - 1][x + 1][0] && 
                        !board[y - 2][x + 2][0] &&
                        !board[y - 3][x + 3][0] &&
                        !board[y - 4][x + 4][0])
                        return 0;
                    if (!board[y][x][1] &&
                        !board[y - 1][x + 1][1] && 
                        !board[y - 2][x + 2][1] &&
                        !board[y - 3][x + 3][1] &&
                        !board[y - 4][x + 4][1])
                        return 0;
                }
            }
            // vertical case:
            if (y + 4 < SIZE)
            {
                if (!board[y][x][0] &&
                    !board[y + 1][x][0] &&
                    !board[y + 2][x][0] &&
                    !board[y + 3][x][0] &&
                    !board[y + 4][x][0])
                    return 0;
                if (!board[y][x][1] &&
                    !board[y + 1][x][1] &&
                    !board[y + 2][x][1] &&
                    !board[y + 3][x][1] &&
                    !board[y + 4][x][1])
                    return 0;
            }
        }
    }
    // if no one wins, it's a draw
    return 1;
}

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
    if (PyArray_AsCArray(&list3_obj, (void ***)&board, dims, 3, descr) < 0) 
    {
        PyErr_SetString(PyExc_TypeError, "error converting to c array");
        return NULL;
    }

    if (is_draw(board))
    {
        PyArray_Free(list3_obj, (void*)board);
        return Py_BuildValue("i", 3);
    }

    /* The game is not an automatic draw. Thus, check if the game is won or lost or continue playing */
    int y, x;

    y = SIZE;
    while (y--)
    {
        x = SIZE;
        while (x--)
        {
            // horizontal case:
            if (x + 4 < SIZE)
            {
                if (board[y][x][0] &&
                    board[y][x + 1][0] && 
                    board[y][x + 2][0] &&
                    board[y][x + 3][0] &&
                    board[y][x + 4][0])
                {
                    PyArray_Free(list3_obj, (void*)board);
                    return Py_BuildValue("i", 1);
                }

                if (board[y][x][1] &&
                    board[y][x + 1][1] && 
                    board[y][x + 2][1] &&
                    board[y][x + 3][1] &&
                    board[y][x + 4][1])
                {
                    PyArray_Free(list3_obj, (void*)board);
                    return Py_BuildValue("i", 2);
                }
                // diagonal case
                if (y + 4 < SIZE)
                {
                    if (board[y][x][0] &&
                        board[y + 1][x + 1][0] && 
                        board[y + 2][x + 2][0] &&
                        board[y + 3][x + 3][0] &&
                        board[y + 4][x + 4][0])
                    {
                        PyArray_Free(list3_obj, (void*)board);
                        return Py_BuildValue("i", 1);
                    }
                    if (board[y][x][1] &&
                        board[y + 1][x + 1][1] && 
                        board[y + 2][x + 2][1] &&
                        board[y + 3][x + 3][1] &&
                        board[y + 4][x + 4][1])
                    {
                        PyArray_Free(list3_obj, (void*)board);
                        return Py_BuildValue("i", 2);
                    }
                }
                if (y - 4 >= 0)
                {
                    if (board[y][x][0] &&
                        board[y - 1][x + 1][0] && 
                        board[y - 2][x + 2][0] &&
                        board[y - 3][x + 3][0] &&
                        board[y - 4][x + 4][0])
                    {
                        PyArray_Free(list3_obj, (void*)board);
                        return Py_BuildValue("i", 1);
                    }
                    if (board[y][x][1] &&
                        board[y - 1][x + 1][1] && 
                        board[y - 2][x + 2][1] &&
                        board[y - 3][x + 3][1] &&
                        board[y - 4][x + 4][1])
                    {
                        PyArray_Free(list3_obj, (void*)board);
                        return Py_BuildValue("i", 2);
                    }
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
                {
                    PyArray_Free(list3_obj, (void*)board);
                    return Py_BuildValue("i", 1);
                }
                if (board[y][x][1] &&
                board[y + 1][x][1] &&
                board[y + 2][x][1] &&
                board[y + 3][x][1] &&
                board[y + 4][x][1])
                {
                    PyArray_Free(list3_obj, (void*)board);
                    return Py_BuildValue("i", 2);
                }
            }
        }
    }
    PyArray_Free(list3_obj, (void*)board);
    return Py_BuildValue("i", 0);
}

static PyObject* 
move_from_policy2D(PyObject* self, PyObject* args) {
    PyObject *list3_obj;
    double thresh;
    if (!PyArg_ParseTuple(args, "Od", &list3_obj, &thresh))
        return NULL;

    double **policy;

    //Create C arrays from numpy objects:
    int typenum = NPY_DOUBLE;
    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(typenum);
    npy_intp dims[2];
    if (PyArray_AsCArray(&list3_obj, (void **)&policy, dims, 2, descr) < 0) 
    {
        PyErr_SetString(PyExc_TypeError, "error converting to c array");
        return NULL;
    }
    int y = SIZE;
    double cum_tot = 0.0;
    while (y--)
    {
        int x = SIZE;
        while (x--)
        {
            cum_tot += policy[y][x];
            if (cum_tot > thresh)
                return Py_BuildValue("ii", y, x);
        }
    }
    return Py_BuildValue("ii", 0, 0);
}

static PyObject *
move_from_policy1D(PyObject *self, PyObject *args)
{
    PyObject *list3_obj;
    double thresh;
    if (!PyArg_ParseTuple(args, "Od", &list3_obj, &thresh))
        return NULL;

    double *policy;

    //Create C arrays from numpy objects:
    int typenum = NPY_DOUBLE;
    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(typenum);
    npy_intp dims[1];
    if (PyArray_AsCArray(&list3_obj, (void *)&policy, dims, 1, descr) < 0) 
    {
        PyErr_SetString(PyExc_TypeError, "error converting to c array");
        return NULL;
    }
    int i = SIZE * SIZE;
    double cum_tot = 0.0;
    while (i--)
    {
        cum_tot += policy[i];
        if (cum_tot > thresh)
            return Py_BuildValue("ii", i/SIZE, i%SIZE);
    }
    return Py_BuildValue("ii", 0, 0);
}

static PyObject *
Q_plus_U(PyObject *self, PyObject *args)
{
    int N, N_b; // N is N(s,a), N_b is N(s,b)
    double P, c_puct, W; // P is P(s,a), c_puct is a constant determining the level of exploration, W is W(s,a).
    if (!PyArg_ParseTuple(args, "ddiid", &c_puct, &W, &N, &N_b, &P)) // args: c_puct, W, N, N_b, P
        return NULL;
    if (N==0)
        return Py_BuildValue("d", c_puct*P*sqrt((double)N_b));
    return Py_BuildValue("d", (double)W/(double)N + c_puct*P*sqrt((double)N_b)/(1.0+(double)N));
}

static PyObject *
minusQ_plus_U(PyObject *self, PyObject *args)
{
    int N, N_b; // N is N(s,a), N_b is N(s,b)
    double P, c_puct, W; // P is P(s,a), c_puct is a constant determining the level of exploration, W is W(s,a).
    if (!PyArg_ParseTuple(args, "ddiid", &c_puct, &W, &N, &N_b, &P)) // args: c_puct, W, N, N_b, P
        return NULL;
    if (N==0)
        return Py_BuildValue("d", c_puct*P*sqrt((double)N_b));
    return Py_BuildValue("d", c_puct*P*sqrt((double)N_b)/(1.0+(double)N)-(double)W/(double)N);
}

static PyMethodDef myMethods[] = {
    {"is_win", is_win, METH_VARARGS, "Input numpy array with shape=(SIZE,SIZE,2,) and return the index corresponding to the game state in this array [\"Continue Playing\", \"Black won\", \"White won\", \"Draw\"]"},
    {"move_from_policy2D", move_from_policy2D, METH_VARARGS, "Input numpy array shape=(SIZE,SIZE,) of the policy: The probabilities of making the move at (y,x) that MUST sum up to 1.0 and a random value between zero and one. Return a random (y,x) with that probability distribution."},
    {"move_from_policy1D", move_from_policy1D, METH_VARARGS, "Input numpy array shape=(SIZE*SIZE,) of the policy: The probabilities of making the move at (y,x) that MUST sum up to 1.0 and a random value between zero and one. Return a random (y=i/SIZE,x=i%SIZE) with that probability distribution."},
    {"Q_plus_U", Q_plus_U, METH_VARARGS, "Input c_puct, W(s,a), N(s,a), sum_[N(s,b)], P(s,a) and return Q+U"},
    {"minusQ_plus_U", minusQ_plus_U, METH_VARARGS, "Input c_puct, W(s,a), N(s,a), sum_[N(s,b)], P(s,a) and return -Q+U(or Q+U from the opponent's POV)"},
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
