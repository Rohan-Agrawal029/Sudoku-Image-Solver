#include<stdio.h>
#include<conio.h>
#include<stdlib.h>

int Solve(int grid[9][9])
{
    int row, col;
    int num;
    for(row=0;row<9;row++)
    {
        for(col=0;col<9;col++)
            if(grid[row][col]==0)
                break;
        if(col<9)
            if(grid[row][col]==0)
                break;
    }
    if(row==9)
        return 1;
    for(num=1;num<=9;num++)
    {
        if(is_valid(grid,row,col,num)==1)
        {
            grid[row][col]=num;
            if(Solve(grid)==1)
                return 1;
            grid[row][col]=0;
        }
    }
    return 0;
}

int is_valid(int grid[9][9], int row, int col, int num)
{
    if(check_row(grid,row,num)==1 && check_col(grid,col,num)==1 && check_box(grid,row,col,num)==1 && grid[row][col]==0)
        return 1;
    return 0;
}

int check_row(int grid[9][9], int row, int num)
{
    int col;
    for(col=0;col<9;col++)
        if(grid[row][col]==num)
            return 0;
    return 1;
}

int check_col(int grid[9][9], int col, int num)
{
    int row;
    for(row=0;row<9;row++)
        if(grid[row][col]==num)
            return 0;
    return 1;
}

int check_box(int grid[9][9], int row, int col, int num)
{
    int boxstartrow, boxstartcol;
    boxstartrow=row-row%3;
    boxstartcol=col-col%3;
    for(row=0;row<3;row++)
        for(col=0;col<3;col++)
            if(grid[row+boxstartrow][col+boxstartcol]==num)
                return 0;
    return 1;
}

void print_grid(int grid[9][9])
{
	int i,j;
	printf("\n");
	for(i=0;i<9;i++)
	{
		if(i%3==0)
			printf("-------------------------\n");
		for(j=0;j<9;j++)
		{
			if(j%3==0)
				printf("| ");
			printf("%d ",grid[i][j]);
		}
		printf("|\n");
	}
	printf("-------------------------\n");
}

void get_grid(int grid[9][9])
{
    int i,j;
    for(i=0;i<9;i++)
        for(j=0;j<9;j++)
            scanf("%d",&grid[i][j]);
}

void make_grid(int grid[9][9], int argc, char *argv[])
{
	int i, j, temp;
	int index = 1;
	for(i=0;i<9;i++)
	{
		for(j=0;j<9;j++)
		{
			temp =atoi(argv[index]);
			grid[i][j] = temp;
			index++;
		}
	}
}	

int main(int argc, char *argv[])
{
    int i,j;
    int grid[9][9];//={{3, 0, 6, 5, 0, 8, 4, 0, 0},{5, 2, 0, 0, 0, 0, 0, 0, 0},{0, 8, 7, 0, 0, 0, 0, 3, 1},{0, 0, 3, 0, 1, 0, 0, 8, 0},{9, 0, 0, 8, 6, 3, 0, 0, 5},{0, 5, 0, 0, 9, 0, 6, 0, 0},{1, 3, 0, 0, 0, 0, 2, 5, 0},{0, 0, 0, 0, 0, 0, 0, 7, 4},{0, 0, 5, 2, 0, 6, 3, 0, 0}};
	/*
	printf("No. of arguments : %d\n", argc);
	for(i=0;i<argc;i++)
	{
		printf("%s\t", argv[i]);
	}
	*/
    if(argc > 1)
	{	
		make_grid(grid, argc, argv);
		//print_grid(grid);
	}
	else
	{
		get_grid(grid);
	}
	if(Solve(grid)==1)
		print_grid(grid);
	else
		printf("\nNo Solution");
    getch();
}








