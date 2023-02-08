#include "ds.h"
#include <stdio.h>
#include <float.h>
#include <inttypes.h>

// Y Cb Cr
double		ColorQuant[NUM_BINS][3] = { {16,145.338745,130.35675}, {16,232.913361,110.241875}, {16,159.62886,113.335052}, {16,116.436996,161.139404}, {16,191.701538,117.329231}, {16,122.648651,139.378372}, {16,173.231567,134.234985}, {16,123.987244,115.811806}, {144,75.237114,171.206192}, {144,97.367981,192.967697}, {144,74.389954,144.218903}, {144,145.439423,108.722794}, {144,101.211136,121.337814}, {144,126.956009,135.119095}, {144,104.02697,158.534225}, {144,175.581299,85.562805}, {80,99.153519,120.484306}, {80,90.457397,155.219727}, {80,191.679169,161.522919}, {80,179.630203,87.103127}, {80,124.204079,150.483521}, {80,155.035553,118.02713}, {80,131.452118,106.519249}, {80,100.848488,188.258133}, {208,16.5,158.320007}, {208,93.840477,137.627792}, {208,86.498314,161.25}, {208,112.241837,152.784866}, {208,64.122246,146.410828}, {208,132.158478,129.667084}, {208,144.561203,105.438805}, {208,106.067795,115.644066}, {48,103.464394,186.472137}, {48,104.118896,134.20401}, {48,186.751877,152.651627}, {48,203.301697,99.239548}, {48,113.30777,158.490158}, {48,115.372551,110.165115}, {48,141.001846,133.632401}, {48,161.815826,105.376053}, {176,101.22065,120.176117}, {176,55.446564,170.893127}, {176,71.792206,141.844162}, {176,99.270424,179.252106}, {176,86.5877,158.480362}, {176,158.94017,97.337608}, {176,134.888199,116.797226}, {176,115.911545,146.571732}, {112,76.475319,146.482376}, {112,147.214554,111.342636}, {112,100.020294,145.749664}, {112,179.014664,87.226059}, {112,89.045067,173.908798}, {112,126.587914,151.940659}, {112,106.692589,119.621223}, {112,110.508881,194.33757}, {240,12.102564,146.641022}, {240,102.453514,120.362808}, {240,70.724808,140.379852}, {240,112.050362,139.103119}, {240,91.427086,140.304688}, {240,134.297729,109.581818}, {240,43.7551,141.857147}, {240,130.689453,129.478516} };

int quantized_color_index(unsigned char *yuvBuff)
{
	int		i, min_index;
	double	dist, min;

	min = DBL_MAX;
	for(i=0; i<NUM_BINS; i++)
	{
		dist = 0;
		dist += (yuvBuff[0] - ColorQuant[i][0]) * (yuvBuff[0] - ColorQuant[i][0]);
		dist += (yuvBuff[1] - ColorQuant[i][1]) * (yuvBuff[1] - ColorQuant[i][1]);
		dist += (yuvBuff[2] - ColorQuant[i][2]) * (yuvBuff[2] - ColorQuant[i][2]);
	
		if( dist < min )
		{
			min = dist;
			min_index = i;
		}
	}

	return min_index;
}

int ColorHistogram(double *HistogramValue, unsigned char *yuvBuff, int width, int height, unsigned char *mask)
{
	int		i, j, k, jj, kk, numpix;
	int		total3, width3;
	width3 = 3 * width;
	total3 = width3 * height;

	for(i=0; i<NUM_BINS; i++)
		HistogramValue[i] = 0;

	numpix = 0;
	for(j=0, jj=0; j<total3; j+=width3, jj+=width)
		for(k=0, kk=0; k<width3; k+=3, kk++)
			// remove background ("white" color in this case)
			if( mask[jj+kk] < 255 )		// if background, mask==255
			{
				HistogramValue[ quantized_color_index(yuvBuff+j+k) ] ++;
				numpix ++;
			}

	if( numpix > 0 )
	{
		for(i=0; i<NUM_BINS; i++)
			HistogramValue[i] = HistogramValue[i] / numpix;
		return 1;
	}
	else
		return 0;		// nothing in this 2D shape
}


typedef struct qdata_{
	int				posX, posY, sizeX, sizeY;
	unsigned char	flag;		// flag=0 ---> spilt left-right, flag=1 ---> spilt top-down
}qdata;


void AddQueue(qdata *Queue, int *qrear, int posX, int posY, int sizeX, int sizeY, int flag)
{
	Queue[*qrear].posX = posX;
	Queue[*qrear].posY = posY;
	Queue[*qrear].sizeX = sizeX;
	Queue[*qrear].sizeY = sizeY;
	Queue[*qrear].flag = flag;
	(*qrear) ++;
}

qdata *DeleteQueue(qdata *Queue, int *qfront, int qrear)
{
	if( *qfront == qrear )
		return NULL;

	(*qfront) ++;
	return Queue+(*qfront-1);
}

// right - left >= 0  ---> 1, otherwise 0
// down - top >= 0  ---> 1, otherwise 0
void CompactColor(unsigned char *CompactValue, double *HistogramValue)
{
	qdata		*pop, Queue[NUM_BINS];
	int			qfront = 0, qrear = 0;
	int			width = 8;		// 8x8 = 64
	int			i, j, jj, count;
	double		left, right;

	AddQueue(Queue, &qrear, 0, 0, 8, 8, 0);

	count = 0;
	while( (pop=DeleteQueue(Queue, &qfront, qrear)) )
	{
		left = right = 0;			// left also denote top, right also denote down
		if( pop->flag == 0 )		// right - left
		{
			for(j=pop->posY, jj=pop->posY*width; j<pop->posY+pop->sizeY; j++, jj+=width)
				for(i=pop->posX; i<pop->posX+pop->sizeX/2; i++)
					left += HistogramValue[jj+i];
			for(j=pop->posY, jj=pop->posY*width; j<pop->posY+pop->sizeY; j++, jj+=width)
				for(i=pop->posX+pop->sizeX/2; i<pop->posX+pop->sizeX; i++)
					right += HistogramValue[jj+i];
		}
		else						// down - top
		{
			for(j=pop->posY, jj=pop->posY*width; j<pop->posY+pop->sizeY/2; j++, jj+=width)
				for(i=pop->posX; i<pop->posX+pop->sizeX; i++)
					left += HistogramValue[jj+i];
			for(j=pop->posY+pop->sizeY/2, jj=(pop->posY+pop->sizeY/2)*width; j<pop->posY+pop->sizeY; j++, jj+=width)
				for(i=pop->posX; i<pop->posX+pop->sizeX; i++)
					right += HistogramValue[jj+i];
		}

		CompactValue[count] = (right>left) ? 1 : 0;
		count ++;

		// push into queue
		if( pop->flag == 0 && pop->sizeX > 1 )
		{
			AddQueue(Queue, &qrear, pop->posX, pop->posY, pop->sizeX/2, pop->sizeY, 1);
			AddQueue(Queue, &qrear, pop->posX+pop->sizeX/2, pop->posY, pop->sizeX/2, pop->sizeY, 1);
		}
		else if( pop->flag == 1 && pop->sizeY > 2 )
		{
			AddQueue(Queue, &qrear, pop->posX, pop->posY, pop->sizeX, pop->sizeY/2, 0);
			AddQueue(Queue, &qrear, pop->posX, pop->posY+pop->sizeY/2, pop->sizeX, pop->sizeY/2, 0);
		}
	}
}

void ExtractCCD(unsigned char *YuvBuff, uint64_t *CCD, unsigned char *SrcBuff)
{
	double			HistogramValue[NUM_BINS];
	unsigned char	CompactValue[NUM_BINS];	// 0 or 1
	int				i;

	// also using "SrcBuff" for checking if there are nothing after rendering
	if( !ColorHistogram(HistogramValue, YuvBuff, WIDTH, HEIGHT, SrcBuff) )
	{
		*CCD = 1;	// use last bit for knowing the 2D shape render nothing
		return ;
	}


//	for(i=0; i<NUM_BINS; i++)
//		printf("%f\n", HistogramValue[i]);
	CompactColor(CompactValue, HistogramValue);		// return CompactValue[0]~[62], total 63 bits
	// CompactValue[NUM_BINS-1] = 0;

	*CCD = 0;
	for(i=0; i<NUM_BINS-1; i++)
	{
		*CCD += CompactValue[i];
		*CCD = (*CCD) << 1;
	}
	// the last bit of "CCD" is 0 for successful extracting
}

double ColorDistance(uint64_t *dest, uint64_t *src)
{
	uint64_t	tmp;
	int					count, i;

	tmp = *dest ^ *src;
	count = 0;
	for(i=0; i<NUM_BINS; i++)
		if( tmp & (0x8000000000000000 >> i) )
			count ++;

	return count;
}

