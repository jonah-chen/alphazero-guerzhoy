#include <stdio.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define SIZE 8

/* 
Create an 8x8 board with
 * 0: empty square
 * 1: black stone
 * 2: white stone
 */

static unsigned char board[SIZE][SIZE];

// Converts signed char in board to meaningful character
static char convert(unsigned char c)
{
    if (c == 0)
        return ' ';
    if (c == 1)
        return 'b';
    if (c == 2)
        return 'w';
    return 'e';
}

static int Cis_win(void)
{
    char y = SIZE;
    char x = SIZE;
    unsigned char draw = 1;
    while (y--)
    {
        while (x--)
        {
            if (board[y][x])
                draw = 0;
            // horizontal case:
            if (x + 4 < SIZE)
            {
                if (board[y][x] == 1 &&
                    board[y][x + 1] == 1 && 
                    board[y][x + 2] == 1 &&
                    board[y][x + 3] == 1 &&
                    board[y][x + 4] == 1)
                    return 1;
                if (board[y][x] == 2 &&
                    board[y][x + 1] == 2 && 
                    board[y][x + 2] == 2 &&
                    board[y][x + 3] == 2 &&
                    board[y][x + 4] == 2)
                    return 2;
                
                if (y + 4 < SIZE)
                {
                    if (board[y][x] == 1 &&
                        board[y + 1][x + 1] == 1 && 
                        board[y + 2][x + 2] == 1 &&
                        board[y + 3][x + 3] == 1 &&
                        board[y + 4][x + 4] == 1)
                        return 1;
                    if (board[y][x] == 2 &&
                        board[y + 1][x + 1] == 2 && 
                        board[y + 2][x + 2] == 2 &&
                        board[y + 3][x + 3] == 2 &&
                        board[y + 4][x + 4] == 2)
                        return 2;
                }
                if (y - 4 >= 0)
                {
                    if (board[y][x] == 1 &&
                        board[y - 1][x + 1] == 1 && 
                        board[y - 2][x + 2] == 1 &&
                        board[y - 3][x + 3] == 1 &&
                        board[y - 4][x + 4] == 1)
                        return 1;
                    if (board[y][x] == 2 &&
                        board[y - 1][x + 1] == 2 && 
                        board[y - 2][x + 2] == 2 &&
                        board[y - 3][x + 3] == 2 &&
                        board[y - 4][x + 4] == 2)
                        return 2;
                }
            }
            // vertical case:
            if (y + 4 < SIZE)
            {
                if (board[y][x] == 1 &&
                board[y + 1][x] == 1 &&
                board[y + 2][x] == 1 &&
                board[y + 3][x] == 1 &&
                board[y + 4][x] == 1)
                    return 1;
                if (board[y][x] == 2 &&
                board[y + 1][x] == 2 &&
                board[y + 2][x] == 2 &&
                board[y + 3][x] == 2 &&
                board[y + 4][x] == 2)
                    return 2;
            }
        }
    }
    if (draw)
        return 3;
    return 0;
}
// Initializes the game by setting all board squares to empty
static PyObject *
init(PyObject *self)
{
    char y = SIZE;
    while (y--)
    {
        char x = SIZE;
        while (x--)
            board[y][x] = 0;
    }
    return Py_BuildValue("i", 0);
}

// DEBUG method: prints the board in readable format
static PyObject *
print_board(PyObject *self)
{
    printf("  0");
    for (char y = 1; y < SIZE; ++y)
        printf("|%d", y);
    
    for (char y = 0; y < SIZE; ++y)
    {
        printf("\n%d|", y);
        for (char x = 0; x < SIZE; ++x)
        {
            printf("%c ", convert(board[y][x]));
        }
    }
    return Py_BuildValue("i", 0);
}

// Makes a move for player player
static PyObject *
move(PyObject *self, PyObject *args)
{
    unsigned char y, x, player;
    if (!PyArg_ParseTuple(args, "bbb", &y, &x, &player))
        return NULL;
    
    // If the board square is already occupied, return error value of -1
    if (board[y][x] || player != 1 && player != 2)
        return Py_BuildValue("i", -1);
    board[y][x] = player;
    return Py_BuildValue("i", 0);
}

// Dangerous force move function.
static PyObject * 
force_move(PyObject *self, PyObject *args)
{
    unsigned char y, x, player;
    if (!PyArg_ParseTuple(args, "bbb", &y, &x, &player))
        return NULL;
    
    board[y][x] = player;
    return Py_BuildValue("i", 0);
}

/* Return the state of the game
 * 0: Contiunue Playing
 * 1: Player 1(Black) Wins
 * 2: Player 2(White) Wins
 * 3: Draw
 */ 

static PyObject *
is_win(PyObject *self)
{
    return Py_BuildValue("i", Cis_win());
}

// Return the version of the program

static PyObject *
version(PyObject *self)
{
    return Py_BuildValue("s", "Version 0.0");
}

static PyMethodDef myMethods[] = {
    {"init", (PyCFunction) init, METH_NOARGS, "Initialize the game by setting all board squares to empty(0)."},
    {"print_board", (PyCFunction) print_board, METH_NOARGS, "DEBUG: Prints the board in human-readable notation."},
    {"move", move, METH_VARARGS, "Make a move for player p at (y,x)=(y,x). Safe Method: If attempts to override a previous move, the move will not be made and returns -1."},
    {"force_move", force_move, METH_VARARGS, "DEBUG: Change position (y,x) on the board to p. WARNING: CAN CAUSE UNEXPECTED ISSUE."},
    {"is_win", (PyCFunction) is_win, METH_NOARGS, "Return the index corresponding to the game state in this array [\"Continue Playing\", \"Black won\", \"White won\", \"Draw\"]"},
    {"version", (PyCFunction) version, METH_NOARGS, "Return the version number."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef myModule = {
    PyModuleDef_HEAD_INIT, 
    "trainGomoku", 
    "A faster implementation of Gomoku game written in C used for training.",
    -1,
    myMethods
};

PyMODINIT_FUNC PyInit_trainGomoku(void)
{
    return PyModule_Create(&myModule);
}
