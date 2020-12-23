#include <stdio.h>

void free_mem(int** x)
{
    for (int i = 0; i < 10; ++i)
    {
        free(x[i]);
    }
    free(x);
}

int main(void)
{
    int **a;
    a = (int**)malloc(10*sizeof(int));
    for (int i = 0; i < 10; ++i)
    {
        a[i] = (int*)malloc(10 * sizeof(int));
    }
    int *c = a[0];

    
    printf("%d\n", c[0]);
    free(a);
}