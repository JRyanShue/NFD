#include <math.h>
#include <malloc.h>
#include <memory.h>

typedef struct POINT_
{
	int x, y;
}POINT;

int MaskValue(unsigned char *r, POINT ImageSize, int j, int k, int l, int MaskNum, POINT *MaskCoor, float *MaskWeight)
{
	int		i;
	float	tmp;

	tmp = 0.0f;
	for(i=0; i<MaskNum ; i++)
	{
		if( k+(MaskCoor+i)->x>=0 && k+(MaskCoor+i)->x<ImageSize.x &&
			l+(MaskCoor+i)->y>=0 && l+(MaskCoor+i)->y<ImageSize.y )
			tmp += *(MaskWeight+i) * *(r+(l+(MaskCoor+i)->y)*ImageSize.x+k+(MaskCoor+i)->x);
		else
			tmp += *(MaskWeight+i) * *(r+j+k);
	}

	return (int)(tmp+0.5);
}

void GRAYSCALEEdgeSobel(unsigned char *dest, unsigned char *src, POINT ImageSize)
{
	int		TotalSize = ImageSize.x * ImageSize.y;
	int		j, k, l;
	int		x, y;
	int		tmp;
	//			-1     1 
	//			-2 < > 2
	//			-1     1
	POINT MaskCoor1[6]={ 
		{-1,-1} , {-1,1} , 
		{0,-1}  , {0,1} , 
		{1,-1}  , {1,1} 
	};
	float MaskWeight1[6] = {-1.0f,1.0f,-2.0f,2.0f,-1.0f,1.0f};
	//			-1 -2 -1 
	//			   < >  
	//			 1  2  1
	POINT MaskCoor2[6]={ 
		{-1,-1} , {-1,0} , {-1,1} , 
		{1,-1} , {1,0} , {1,1} 
	};
	float MaskWeight2[6] = {-1.0f,-2.0f,-1.0f,1.0f,2.0f,1.0f};

	for(j=0, l=0; j<TotalSize; j+=ImageSize.x, l++)
		for(k=0; k<ImageSize.x; k++)
		{
			x = MaskValue(src, ImageSize, j, k, l, 6, MaskCoor1, MaskWeight1);
			y = MaskValue(src, ImageSize, j, k, l, 6, MaskCoor2, MaskWeight2);
			tmp = (int)sqrt( x*x + y*y );
			if(tmp>255)
				tmp = 255;
			*(dest+j+k) = tmp;
		}
}

// test edge from depth
void EdgeDetect(unsigned char *dest, unsigned char *src, int width, int height)
{
	POINT	ImageSize;
	ImageSize.x = width;
	ImageSize.y = height;
	GRAYSCALEEdgeSobel(dest, src, ImageSize);
}

// test edge from silhouette
void EdgeDetectSil(unsigned char *edge, unsigned char *src, int width, int height)
{
	int		TotalSize = width * height;
	int		j, k, l;

	for(j=0, l=0; j<TotalSize; j+=width, l++)	// y
		for(k=0; k<width; k++)					// x
			if( *(src+j+k) < 255 )	// inside
			{
				// edge only
				*(edge+j+k) = 255;
				// enhence edge
//				*(src+j+k) = 128;
				if( l>0 && *(src+j+k-width) == 255 )
					*(edge+j+k) = 0;		// edge
				if( k>0 && *(src+j+k-1) == 255 )
					*(edge+j+k) = 0;		// edge
				if( l<height-1 && *(src+j+k+width) == 255 )
					*(edge+j+k) = 0;		// edge
				if( k<width-1 && *(src+j+k+1) == 255 )
					*(edge+j+k) = 0;		// edge

				if( l>0 && k>0 && *(src+j+k-width-1) == 255 )
					*(edge+j+k) = 0;		// edge
				if( l>0 && k<width-1 && *(src+j+k-width+1) == 255 )
					*(edge+j+k) = 0;		// edge
				if( l<height-1 && k>0 && *(src+j+k+width-1) == 255 )
					*(edge+j+k) = 0;		// edge
				if( l<height-1 && k<width-1 && *(src+j+k+width+1) == 255 )
					*(edge+j+k) = 0;		// edge
			}
			else						// background
				*(edge+j+k) = 255;
}

