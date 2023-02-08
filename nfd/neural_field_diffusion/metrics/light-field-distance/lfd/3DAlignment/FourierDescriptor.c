#include <stdio.h>
#include <malloc.h>
#include <memory.h>

#include "fftw/rfftw.h"

#include "ds.h"
#include "TraceContour.h"
#include "Thin.h"
#include "Bitmap.h"
#include "Morphology.h"

int IsMultiPart(unsigned char *ContourMask, unsigned char *Y, int width, int height)
{
	int		insideArea, outsideArea;
	int		i, j, k, change, total;
	int		MaskCoor[4] = {-1, 1, -width, width};

	total = width * height;

	// inside: 0; contour: 255; outside: 128
	for(i=0; i<width; i++)
		ContourMask[i] = 128;
	for(i=total-width; i<total; i++)
		ContourMask[i] = 128;
	for(i=0; i<total; i+=width)
		ContourMask[i] = ContourMask[i+width-1] = 128;
	
	do
	{
		change = 0;

		// left-right, top-bottom
		for(j=width; j<total-width; j+=width)
			for(k=1; k<width-1; k++)
				if( ContourMask[j+k] == 0 )
					for(i=0; i<4; i++)
						if( ContourMask[j+k+MaskCoor[i]] == 128 )
						{
							ContourMask[j+k] = 128;
							change = 1;
						}
		// right-left, bottom-top
		for(j=total-2-width; j>0; j-=width)
			for(k=width-1; k>0; k--)
				if( ContourMask[j+k] == 0 )
					for(i=0; i<4; i++)
						if( ContourMask[j+k+MaskCoor[i]] == 128 )
						{
							ContourMask[j+k] = 128;
							change = 1;
						}

	}while( change );

//WriteBitmap8(ContourMask, width, height, "ttt2.bmp");

	// get area of contour area
	insideArea = outsideArea = 0;
	for(i=0; i<total; i++)
		if( Y[i] < 255 )
		{
			if( ContourMask[i] == 128 )
				outsideArea ++;
			else
				insideArea ++;
		}

	return ( (double)insideArea / (double)(insideArea+outsideArea) > 0.95 ) ? 0 : 1;
}

// *Y: 255 is background
// input is an edge image
void FourierDescriptor(double FdCoeff[], unsigned char *Y, int width, int height, 
					   sPOINT *Contour, unsigned char *ContourMask, double CenX, double CenY)
{
	int				total, i, k, N;//, cenx, ceny;
	fftw_real		*CenDist;	// the contour is the input of fourier descriptor
    rfftw_plan		p;
	fftw_real		*out, *power_spectrum;
	int				num;
	unsigned char	*Buff;
	unsigned char	*flag;	// use for another mask in Thin()
	POINT			Size={width, height};
//FILE *fpt;
//fpt = fopen("testfd.txt", "w");
//for(i=0; i<width*height; i++)
//	fprintf(fpt, "%d,", Y[i]);
//fclose(fpt);

	total = width * height;

//WriteBitmap8(Y, width, height, "tt1.bmp");
	num = TraceContour(Contour, ContourMask, Y, width, height);

//fpt = fopen("testtc.txt", "w");
//fprintf(fpt, "%d\n", num);
//for(i=0; i<num; i++)
//	fprintf(fpt, "%d\n", Contour[i].y*width+Contour[i].x);
//fclose(fpt);

//WriteBitmap8(ContourMask, width, height, "tt2.bmp");
	if( IsMultiPart(ContourMask, Y, width, height) )
	{
		Buff = (unsigned char*) malloc( total * sizeof(unsigned char));
		memcpy(Buff, Y, total * sizeof(unsigned char));
		BINARYErosion3x3(Buff, Size);
		// fix bug: IsMultiPart() should be run after TraceContour()
		TraceContour(Contour, ContourMask, Buff, width, height);
		if( IsMultiPart(ContourMask, Buff, width, height) )	// fix bug: should be "Buff" NOT "Y"
		{
			BINARYErosion5x5(Buff, Size);
			TraceContour(Contour, ContourMask, Buff, width, height);
			if( IsMultiPart(ContourMask, Buff, width, height) )	// fix bug: should be "Buff" NOT "Y"
			{
				memcpy(Buff, Y, total * sizeof(unsigned char));
				BoundingBox(Buff, Size);
			}
		}

		flag = (unsigned char *) malloc( total * sizeof(unsigned char));
//WriteBitmap8(Y, width, height, "t3.bmp");
		Thin(Buff, Y, flag, width, width*height);
//WriteBitmap8(Buff, width, height, "t4.bmp");

		num = TraceContour(Contour, ContourMask, Buff, width, height);
		free(Buff);
		free(flag);
	}
//WriteBitmap8(ContourMask, width, height, "ttt_1.bmp");

	if( num < 8 )
	{
//		printf("error!!\n");
		for(i=0; i<FD_COEFF_NO; i++)
			FdCoeff[i] = 0;
		return ;
	}

	CenDist = (fftw_real *) malloc( num * sizeof(fftw_real));
//	cenx = width / 2;
//	ceny = height / 2;
	for(i=0; i<num; i++)
//		CenDist[i] = (fftw_real) HYPOT(Contour[i].x-cenx, Contour[i].y-ceny);
//		CenDist[i] = (fftw_real) HYPOT(Contour[i].x-CENTER_X, Contour[i].y-CENTER_Y);
		CenDist[i] = (fftw_real) HYPOT(Contour[i].x-CenX, Contour[i].y-CenY);

//fpt = fopen("testCenDist.txt", "w");
//for(i=0; i<num; i++)
//	fprintf(fpt, "%f\n", CenDist[i]);
//fclose(fpt);

	// fft, get fourier descriptor from the contour
	N = num;
	out = (fftw_real *) malloc(N * sizeof(fftw_real));
	power_spectrum = (fftw_real *) malloc( (N/2+1) * sizeof(fftw_real));

	p = rfftw_create_plan(N, FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE);
	rfftw_one(p, CenDist, out);

	// fix bug: The image part should be 0, not out[0]
	power_spectrum[0] = HYPOT(out[0],0);  // out[0]*out[0];  // DC component
	for (k = 1; k < (N+1)/2; ++k)  // (k < N/2 rounded up)
		power_spectrum[k] = HYPOT(out[k], out[N-k]);	// out[k]*out[k] + out[N-k]*out[N-k];
	if (N % 2 == 0) // N is even
		// fix bug: The image part should be 0, not out[N/2]
		power_spectrum[N/2] = HYPOT(out[N/2], 0);	// out[N/2]*out[N/2]; // Nyquist freq.

	rfftw_destroy_plan(p);

	N= N/2+1;		// fix bug: power_spectrum[0~N/2] only
	for(i=1; i<=FD_COEFF_NO; i++)
		if( i<N )
			FdCoeff[i-1] = power_spectrum[i]/power_spectrum[0];
		else
			FdCoeff[i-1] = 0;

//fpt = fopen("testps.txt", "w");
//for(i=0; i<(N+1)/2; i++)
//	fprintf(fpt, "%f\n", power_spectrum[i]);
//fclose(fpt);

//fpt = fopen("testfft.txt", "w");
//for(i=0; i<num; i++)
//	fprintf(fpt, "%f\n", out[i]);
//fclose(fpt);

//fpt = fopen("testfdcoeff.txt", "w");
//for(i=0; i<FD_COEFF_NO; i++)
//	fprintf(fpt, "%f\n", FdCoeff[i]);
//fclose(fpt);

	free(out);
	free(power_spectrum);
	free(CenDist);
	return;
}


/*
#include "rfftw.h"

#define N 10

void main()
{
	fftw_complex in[N]={{10,0},{8,0},{6,0},{4,0},{2,0},{4,0},{8,0},{12,0},{6,0},{0,0}}, *out;
	fftw_plan p;
	int	i;

	out = (fftw_complex *) malloc( N * sizeof(fftw_complex));

	p = fftw_create_plan(N, FFTW_FORWARD, FFTW_ESTIMATE);

	fftw_one(p, in, out);

	fftw_destroy_plan(p); 

	for(i=0; i<N; i++)
		printf("(%.3f, %.3f) ", out[i].re, out[i].im);
	printf("\n");

	for(i=0; i<N; i++)
		printf("%.3f ", out[i].re*out[i].re+out[i].im*out[i].im);
	printf("\n");

	free(out);
}
*/

/*
#define N 10

#include "rfftw.h"
void main()
{
	fftw_real in[N]={10,8,6,4,2,4,8,12,6,0}, out[N], power_spectrum[N/2+1];
     rfftw_plan p;
     int i, k;

     p = rfftw_create_plan(N, FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE);

     rfftw_one(p, in, out);
     power_spectrum[0] = out[0]*out[0];  // DC component
     for (k = 1; k < (N+1)/2; ++k)  // (k < N/2 rounded up)
          power_spectrum[k] = out[k]*out[k] + out[N-k]*out[N-k];
     if (N % 2 == 0) // N is even
          power_spectrum[N/2] = out[N/2]*out[N/2];  // Nyquist freq.

     rfftw_destroy_plan(p);

	for(i=0; i<N; i++)
		printf("%.3f ", out[i]);
	printf("\n");

	for(i=0; i<N/2+1; i++)
		printf("%.3f ", power_spectrum[i]);
	printf("\n");

}
*/
