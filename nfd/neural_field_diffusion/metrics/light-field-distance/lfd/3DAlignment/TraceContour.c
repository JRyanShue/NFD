#include <stdio.h>
#include <memory.h>
#include "Bitmap.h"
#include "ds.h"

int GetStart(unsigned char *Y, int width, int height)
{
	unsigned char		*pY, *total = Y + width * height;

	for(pY=Y; pY < total; pY++)
		if( *pY < 255 )
			return pY-Y;

	return -1;
}

// input is a 1D array
// the boundary of the input image should be white (background) to avoid overflow
int TraceContour(sPOINT *Contour, unsigned char *ContourMask, unsigned char *Y, int width, int height)
{
	//	dirP[0~3] denote down, right, up, left
	//	dirP[x][0~2] denote P1, P2 and P3
	int		nextPos[4][3] = {	{width-1, width, width+1}, {width+1, 1, -width+1}, 
								{-width+1, -width, -width-1}, {-width-1, -1, width-1}};
	int		nextDir[4][3] = { {3,0,0}, {0,1,1}, {1,2,2}, {2,3,3}};

	int		curPos, curDir;
	int		i, j, start, walk, mayLoss;
	int		count;

	if( (start = GetStart(Y, width, height)) < 0 )
		return -1;		// error, no pixel exists

	// there are three case may miss some part (b: background; s: start point; v: foreground)
	// b b b    b b b    b b b
	// b s b    b s v    b s v
	// v b v    v b b    v b v
	// and there are no cicle from left 'v' to right 'v'
	// set mayLoss=1 if in one of the three case
	if( Y[start+width-1]<255 && Y[start+width]==255 && (Y[start+width+1]<255 || Y[start+1]<255 ) )
		mayLoss = 1;
	else
		mayLoss = 0;

	curPos = start;
	curDir = 0;		// initially, face down

	count = 0;

	Contour[count].x = curPos%width;
	Contour[count].y = curPos/width;
	memset(ContourMask, 0, width*height * sizeof(unsigned char));
	ContourMask[curPos] = 255;	// the pixel is contour
	count ++;

	while(1)
	{
		walk = 0;
		for(j=0; j<4; j++)
		{
			for(i=0; i<3; i++)
				if( Y[ curPos + nextPos[curDir][i] ] < 255 )
				{
					curPos += nextPos[curDir][i];
					curDir = nextDir[curDir][i];

					Contour[count].x = curPos%width;
					Contour[count].y = curPos/width;
					ContourMask[curPos] = 255;	// the pixel is contour
					count ++;

					walk = 1;
					break;
				}

			if(walk == 1)
				break;

			curDir = (curDir+1) % 4 ;
		}

		if( walk == 0 )
			return -1;		// isolated pixel, return error

		// if the three case and the last return is left (curDir=3), then go to right part
		if(curPos == start && ( mayLoss!=1 || curDir!=1 ) )
			return count;		// success
	}
}
