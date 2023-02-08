
unsigned char DelTab[256] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1};
unsigned char IsDel(unsigned char *flag, int coor, int *MaskCoor)
{
	int		k, index;

	index = 0;
	for(k=1; k<9; k++)
		index += flag[coor+MaskCoor[k]] << (k-1);

	return DelTab[index];
}

int DelSide(unsigned char *flag, unsigned char *src, unsigned char *SilhMask, int width, int TotalSize, int *MaskCoor, int side)
{
	int		j, k;
	int		change;

	change = 0;

	// left-right, top-bottom
	for(j=width; j<TotalSize-width; j+=width)
		for(k=1; k<width-1; k++)
			// if the point is border in this side, and is not a simple
			// The deletion of border points from a given side of S should be done "in parallel"
			if( flag[j+k] &&							// the pixel 
				flag[j+k+MaskCoor[side]] == 0 &&		// this side is nothing (so is a border)
				IsDel(flag, j+k, MaskCoor) &&
				SilhMask[j+k]==255 )						// if the pixel is background
			{
				src[j+k] = 255;		// set to be background
				change = 1;
			}

	// "flag" is also used for recording which pixel is foreground before doing the above loop
	if( change )
	{
		// left-right, top-bottom
		for(j=width; j<TotalSize-width; j+=width)
			for(k=1; k<width-1; k++)
				if( flag[j+k] && src[j+k] == 255 )
					flag[j+k] = 0;
	}

	return change;
}

// get thin from "src", and save back to "src" and "flag"
void Thin(unsigned char *src, unsigned char *SilhMask, unsigned char *flag, int width, int TotalSize)
{
	int				i, change;
	//				coordinate offset of the neighborhood
	//				7 2 6
	//		        3 0 1
	//			    8 4 5
	int				MaskCoor[9] = {0, 1, -width, -1, width, 1+width, 1-width, -1-width, -1+width};
	int				SideOrder[8] = {2, 4, 3, 1, 7, 5, 8, 6};

	// here assume the boundary is 255, foreground is <255 and >=0
	for(i=0; i<TotalSize; i++)
		flag[i] = (src[i]<255) ? 1 : 0;

	do	{
		change = 0;

		// 4-neighbor
		for(i=0; i<4; i++)
			if( DelSide(flag, src, SilhMask, width, TotalSize, MaskCoor, SideOrder[i]) )
				change = 1;
	}while( change );

}
