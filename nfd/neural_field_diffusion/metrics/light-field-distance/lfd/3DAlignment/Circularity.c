#include "edge.h"
#include "Bitmap.h"

#define		PI4		12.5663704

char fn[100];
int c=0;

double Circularity(unsigned char *srcBuff, int width, int height, unsigned char *edge)
{
	int		TotalSize = width * height;
	int		i, A, p;
	double	cir;

	A = 0;		// area
	for(i=0; i<TotalSize; i++)
		if( srcBuff[i] < 255 )
			A ++;

//sprintf(fn, "%02d.bmp", c++);
//WriteBitmap8(srcBuff, width, height, fn);
//	EdgeDetectSil(srcBuff, width, height);
//sprintf(fn, "%03d.bmp", c++);
//WriteBitmap8(srcBuff, width, height, fn);

	p = 0;		// perimeter
	for(i=0; i<TotalSize; i++)
		if( edge[i] < 255 )
			p ++;

// define in MPEG-7, range [0~110], not good
//return (double)(p*p)/(double)A;

	if( p>0 )
	{
		cir = PI4 * A / ( p * p );
		if( cir > 1 )
			cir = 1;
	}
	else
		cir = 0;		// if render nothing (bad)

	return cir;

}
