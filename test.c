#include <stdio.h>
#define SIZE 8

unsigned char board[SIZE][SIZE];

// Converts signed char in board to meaningful character
static char convert(unsigned char c)
{
    if (c == 0)
        return ' ';
    if (c == 1)
        return 'b';
    if (c == 2)
        return 'w';
    return NULL;
}

// Initializes the game by setting all board squares to empty
static void
init(void)
{
    char y = SIZE;
    while (y--)
    {
        char x = SIZE;
        while (x--)
            board[y][x] = 0;
    }
}

// DEBUG method: prints the board in readable format
static void
print_board(void)
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
}

// Makes a move for player player
static int
move(unsigned char y, unsigned char x, unsigned char player)
{    
    // If the board square is already occupied, return error value of -1
    if (board[y][x] || player != 1 && player != 2)
        return -1;
    board[y][x] = player;
    return 0;
}

// Dangerous force move function.
static void
force_move(unsigned char y, unsigned char x, unsigned char player)
{
    board[y][x] = player;
}

/* Return the state of the game
 * 0: Contiunue Playing
 * 1: Player 1(Black) Wins
 * 2: Player 2(White) Wins
 * 3: Draw
 */ 
static int is_win(void)
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
                if (y - 4 < SIZE)
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

int main(void)
{
    init();
    print_board();

    int turn = 0;
    while(!is_win())
    {
        int y, x;
        scanf("%d %d", &y, &x);
        if (turn++ % 2 == 0)
            force_move((unsigned char) y, (unsigned char) x, 1);
        else 
            force_move((unsigned char) y, (unsigned char) x, 2);
        print_board();
    }
    return 0;   
}