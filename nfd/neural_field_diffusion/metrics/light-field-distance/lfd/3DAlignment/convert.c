#include "ds.h"

void  RGB_To_YUV(unsigned char *yuvBuff, unsigned char *rgbBuff, int width, int height)
{

	int		i, j, total, width3;

	width3 = 3 * width;
	total = width3 * height;
	for(i=0; i<total; i+=width3)
		for(j=0; j<width3; j+=3)
		{
			// Y
			yuvBuff[i+j  ] = (unsigned char) (0.299*rgbBuff[i+j] + 0.587*rgbBuff[i+j+1] + 0.114 * rgbBuff[i+j+2]);
			// U = 0.493(B-Y)
			// Cb = B-Y
			yuvBuff[i+j+1] = 128 + (unsigned char) (-0.16874*rgbBuff[i+j] - 0.33126*rgbBuff[i+j+1] + 0.5 * rgbBuff[i+j+2]);
			// V = 0.877(R-Y)
			// Cr = R-Y
			yuvBuff[i+j+2] = 128 + (unsigned char) (0.5*rgbBuff[i+j] - 0.41869 *rgbBuff[i+j+1] -0.08131 * rgbBuff[i+j+2]);
		}
}

