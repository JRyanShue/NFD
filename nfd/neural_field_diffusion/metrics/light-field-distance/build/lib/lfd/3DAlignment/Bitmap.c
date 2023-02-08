#include <stdio.h>
#include "TypeConverts.h"

// the input "Buff" has "no" pad after every row
// so, it must add "pad" after every row
void WriteBitmap(unsigned char *Buff, int x, int y, char *filename)
{
	BITMAPFILEHEADER	FileHeader;
	BITMAPINFOHEADER	InfoHeader;
	FILE *fpt;
	int i, j;
	BYTE PadData;
	int pad, PadWidth, NoPadWidth, NoPadSize;
	POINT Size;
	Size.x = x;
	Size.y = y;

	PadData = 0;
	PadWidth = 3 * Size.x;
	NoPadWidth = PadWidth;
	NoPadSize = PadWidth * Size.y;
	pad = ( 4 - (PadWidth & 0x03) ) & 0x03;
	PadWidth += pad;

	fpt=fopen(filename, "wb");

	FileHeader.bfType = 19778;
	FileHeader.bfSize = NoPadSize+56;
	FileHeader.bfReserved1 = 0;
	FileHeader.bfReserved2 = 0;
	FileHeader.bfOffBits = 54;

	InfoHeader.biSize = 40;
	InfoHeader.biWidth = Size.x;
	InfoHeader.biHeight = Size.y;
	InfoHeader.biPlanes = 1;
	InfoHeader.biBitCount = 24;
	InfoHeader.biSizeImage = 0;
	InfoHeader.biXPelsPerMeter = 2834;
	InfoHeader.biYPelsPerMeter = 2834;
	InfoHeader.biClrUsed = 0;
	InfoHeader.biCompression = 0;
	fwrite(&FileHeader, sizeof(BITMAPFILEHEADER), 1, fpt);
	fwrite(&InfoHeader, sizeof(BITMAPINFOHEADER), 1, fpt);

	// if pad = 1, 2, or 3, remove pad from every row, and
	// save   rgb,rgb,rgb,rgb,.., from (left,top) to (right,bottom)

	// if no pad, the second "fwrite" write nothing to disk
	if(pad==0)
		fwrite(Buff, NoPadWidth*Size.y, 1, fpt);
	else
	{
		for(i=0, j=0; i<Size.y; i++, j+=NoPadWidth)
		{
			fwrite(Buff+j, NoPadWidth, 1, fpt);
			fwrite(&PadData, pad, 1, fpt);
		}
	}

	fputc(0, fpt);
	fputc(0, fpt);
	fclose(fpt);
}

void WriteBitmap8(unsigned char *sBuff, int x, int y, char *filename)
{
	BITMAPFILEHEADER	FileHeader;
	BITMAPINFOHEADER	InfoHeader;
	FILE *fpt;
	int i, j, k, l;
	BYTE PadData;
	int pad, PadWidth, NoPadWidth, NoPadSize;
	// copy gray data to rgb data
	unsigned char	*Buff;
	Buff = (unsigned char*) malloc( 3 * x * y * sizeof(unsigned char));
	for(i=0, k=0, l=0; i<y; i++)
		for(j=0; j<x; j++, k+=3, l++)
			Buff[k] = Buff[k+1] = Buff[k+2] = sBuff[l];

	PadData = 0;
	PadWidth = 3 * x;
	NoPadWidth = PadWidth;
	NoPadSize = PadWidth * y;
	pad = ( 4 - (PadWidth & 0x03) ) & 0x03;
	PadWidth += pad;

	fpt=fopen(filename, "wb");

	FileHeader.bfType = 19778;
	FileHeader.bfSize = NoPadSize+56;
	FileHeader.bfReserved1 = 0;
	FileHeader.bfReserved2 = 0;
	FileHeader.bfOffBits = 54;

	InfoHeader.biSize = 40;
	InfoHeader.biWidth = x;
	InfoHeader.biHeight = y;
	InfoHeader.biPlanes = 1;
	InfoHeader.biBitCount = 24;
	InfoHeader.biSizeImage = 0;
	InfoHeader.biXPelsPerMeter = 2834;
	InfoHeader.biYPelsPerMeter = 2834;
	InfoHeader.biClrUsed = 0;
	InfoHeader.biCompression = 0;
	fwrite(&FileHeader, sizeof(BITMAPFILEHEADER), 1, fpt);
	fwrite(&InfoHeader, sizeof(BITMAPINFOHEADER), 1, fpt);

	// if pad = 1, 2, or 3, remove pad from every row, and
	// save   rgb,rgb,rgb,rgb,.., from (left,top) to (right,bottom)

	// if no pad, the second "fwrite" write nothing to disk
	if(pad==0)
		fwrite(Buff, NoPadWidth*y, 1, fpt);
	else
	{
		for(i=0, j=0; i<y; i++, j+=NoPadWidth)
		{
			fwrite(Buff+j, NoPadWidth, 1, fpt);
			fwrite(&PadData, pad, 1, fpt);
		}
	}

	fputc(0, fpt);
	fputc(0, fpt);
	fclose(fpt);
	free(Buff);
}

void WriteRAW(unsigned char *Buff, int x, int y, char *filename)
{
	FILE	*fpt;
	int		i;

	fpt=fopen(filename, "wb");
	// write width and height
	fwrite(&x, 4, 1, fpt);
	fwrite(&y, 4, 1, fpt);
	// write only 1 byte per pixel
	for(i=0; i<3*x*y; i+=3)
		fprintf(fpt, "%c", Buff[i]);
	fclose(fpt);
}