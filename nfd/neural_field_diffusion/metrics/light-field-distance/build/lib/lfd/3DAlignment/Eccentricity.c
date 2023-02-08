#include <math.h>
#include "Eccentricity.h"

#define	POW2(a)		((a)*(a))

double Eccentricity(unsigned char *srcBuff, int width, int height, double CenX, double CenY)
{
	int					x, y, count;
	unsigned char		*pImage;
	double				i11, i02, i20;
	double				ecc;

	count = 0;
	pImage = srcBuff;
	i11 = i02 = i20 = 0;
	for (y=0 ; y<height ; y++)
	for (x=0 ; x<width; x++)
	{
		if( *pImage < 255 )
		{
			i02 += POW2(y-CenY);
			i11 += (x-CenX) * (y-CenY);
			i20 += POW2(x-CenX);
			count ++;		// how many pixels
		}
		pImage++;
	}

	if( count > 1 )
	{
		// defined in MPEG-7, seen not good
//		dtmp = sqrt( POW2(i20-i02) + 4 * POW2(i11) );
//		if( (i20+i02-dtmp) > 0 )
//			return sqrt( (i20+i02+dtmp) / (i20+i02-dtmp) );
//		else
//			return 1.0;

		// defined in other paper
		ecc = ( POW2(i20-i02) + 4 * POW2(i11) ) / POW2(i20+i02);
		if( ecc > 1 )
			ecc = 1;
	}
	else
		ecc = 0;

	return ecc;
}
